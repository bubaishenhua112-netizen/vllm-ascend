# D2RHConnector KV Cache 切分组装逻辑分析


### KV Cache Groups 的概念

在 vLLM 中,模型的所有 attention 层并非千篇一律,不同层可能使用不同的 KV cache 格式。vLLM 将 **使用相同 KV cache 格式(相同 spec 类型)的层归为一个 group**。每个 group 拥有:

- `group_id`: group 在 `kv_cache_groups` 列表中的索引
- `kv_cache_spec`: 描述该 group 的 KV cache 格式,可能是:
  - `FullAttentionSpec` / `MLAAttentionSpec` — 标准 attention 或 MLA
  - `MambaSpec` — 状态空间模型(如 Mamba),KV cache 是状态张量而非 head 维度的 KV
  - `SlidingWindowMLASpec` — DSv4 的稀疏滑动窗口 MLA
- `layer_names`: 该 group 包含的所有层名

例如,一个混合架构模型可能有 2 个 group:
- group 0: 前 3 层是 FullAttention (num_kv_heads=8)
- group 1: 后 27 层是 MLA (num_kv_heads=128,但 latent 复制)

**不同 group 的切分策略可以不同**,这正是 `_build_group_pulls_by_port` 对每个 group 独立计算 `num_group_pulls` 的原因。

### 对每个 group 独立计算切分数

```python
# _build_group_pulls_by_port 中,遍历每个 KV cache group
for group_id, group in enumerate(self.kv_cache_groups):
    if isinstance(group.kv_cache_spec, MambaSpec):
        # ★ Mamba:切分数 = P_TP / D_TP(与 attention 相反)
        # Mamba 的 state 是按 TP 切分的(每个 rank 持有部分 state),
        # D rank 需要从多个 P rank 汇聚完整 state
        num_group_pulls = max(1, prefill_tp_size // self._decode_tp_size)
    else:
        # ★★ Attention 切分核心:计算每个 D rank 需要几次 pull 组装完整 head
        group_spec = self._serialize_group_for_scheduler(group)
        num_group_pulls = self._get_attention_group_num_need_pulls(
            group_spec, prefill_tp_size)
    # ★ remote_tp_offset:标识这是第几个切分片(0, 1, ..., num_group_pulls-1)
    remote_tp_offset = rank_idx % num_group_pulls
    group_pulls.append(GroupPull(
        group_id=group_id,
        remote_tp_offset=remote_tp_offset,
        num_group_pulls=num_group_pulls,
        ...
    ))
```

### Attention 的切分数公式

```python
def _get_attention_group_num_need_pulls(self, group_spec, prefill_tp_size):
    # 从 group_spec 中读取该 group 的 KV head 数
    # (不同 group 可能有不同的 num_kv_heads,如 hybrid 模型)
    kv_cache_spec = group_spec.get("kv_cache_spec", {})
    num_key_value_heads = self.num_key_value_heads
    if isinstance(kv_cache_spec, dict):
        for key in ("num_kv_heads", "num_key_value_heads"):
            if isinstance(kv_cache_spec.get(key), int):
                num_key_value_heads = kv_cache_spec[key]
                break

    # ★ D 节点每个 rank 持有的 head 数
    num_d_block_heads = max(1, num_key_value_heads // self.tp_size)
    # ★ P 节点每个 rank 持有的 head 数
    num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
    # ★ 切分数 = D_head / P_head (每个 D rank 需要从几个 P rank 拼装)
    return num_d_block_heads // num_p_block_heads
```

**核心公式**:

```
num_group_pulls = (num_kv_heads // D_TP) // (num_kv_heads // P_TP)
```

- `num_d_block_heads`: D 节点每个 rank 持有的 KV head 数 = `num_kv_heads // D_TP`
- `num_p_block_heads`: P 节点每个 rank 持有的 KV head 数 = `num_kv_heads // P_TP`
- `num_group_pulls`: 每个 D rank 需要从几个 P rank 拉取才能拼满自己的 head 数

**为什么是 D_head / P_head?** 因为普通 attention 中 KV cache 按 head 维度在 TP 间切分。P 节点 TP 更大,每个 P rank 的 head 更少;D 节点 TP 更小,每个 D rank 的 head 更多。D rank 需要从多个 P rank 收集 head 片段来拼满自己的 block。

**约束**: 代码强制要求 `prefill_tp_size >= decode_tp_size`,否则 `num_group_pulls` 会为 0,无法切分。

---

## 字节级切分与传输(Hop1: P NPU → CPU 暂存)

### 切分发生在哪里

`D2RHThread._transfer_kv_cache_all_groups` 负责执行 Hop1 传输。对每个 group 的每个层,它将一个完整的 block 按字节切成 `num_group_pulls` 份,每次只传输其中一份。

```python
# D2RHThread._transfer_kv_cache_all_groups 中
for group_pull in group_pulls:
    group_id = group_pull.group_id
    num_group_pulls = group_pull.num_group_pulls      # 切分数
    remote_tp_offset = group_pull.remote_tp_offset      # 第几个切分片

    # ...映射 block id,展开 kernel block id...

    for layer_idx in layer_indices:
        for cache_idx in range(len(self.cpu_kv_caches_base_addr[layer_idx])):
            src_layer_base_addr = self.cpu_kv_caches_base_addr[layer_idx][cache_idx]
            dst_layer_base_addr = remote_base_addrs[layer_idx][cache_idx]
            block_len = self.cpu_block_len_per_addr[layer_idx][cache_idx]
            block_stride = self.cpu_block_stride_per_addr[layer_idx][cache_idx]
            remote_block_stride = remote_block_stride_per_addr[layer_idx][cache_idx]

            # ★★ 字节级切分:将一个完整 block 切成 num_group_pulls 份
            inner_block_len = block_len // num_group_pulls
            ...
            for remote_block_id, local_block_id in zip(transfer_remote_block_ids,
                                                        transfer_local_block_ids):
                # Hop1: 从 P 节点读 inner_block_len 字节 → CPU 暂存
                src_list.append(
                    src_layer_base_addr + local_block_id[0] * block_stride)
                dst_list.append(
                    dst_layer_base_addr + remote_block_id[0] * remote_block_stride)
                length_list.append(inner_block_len * len(local_block_id))

    # ★ 调用 mooncake engine 执行批量传输
    ret = self.engine.batch_transfer_sync_read(
        session_id, src_list, dst_list, length_list)
```

### Hop2 组装:CPU 暂存 → D NPU

Hop1 只把每个 P rank 的 head 片段拉到独立的 CPU 暂存块。Hop2 在 `KVCacheRecvingThread`(基类 `_transfer_kv_cache_all_groups`)中,将这些片段 **按 `remote_tp_offset` 偏移写入 D NPU block 的对应位置**,完成拼装:

```python
# 基类 KVCacheRecvingThread._transfer_kv_cache_all_groups 中 (mooncake_connector.py:769-790)
for group_pull in group_pulls:
    tp_num_need_pulls = group_pull.num_group_pulls
    inner_offset = group_pull.remote_tp_offset   # ★ 切分偏移
    ...
    inner_block_len = block_len // tp_num_need_pulls
    ...
    for remote_block_id, local_block_id in zip(...):
        # ★★ Hop2 关键:在 D NPU 目标地址上加偏移,拼装完整 block
        src = src_layer_base_addr + local_block_id[0] * block_stride \
              + inner_offset * inner_block_len   # ← 偏移写入位置
        dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
        length = inner_block_len * len(local_block_id)
        src_list.append(src)
        dst_list.append(dst)
        length_list.append(length)
```

### 两跳协作示意图

```
                    inner_block_len
P rank 0:  [====头0头1====]                    P rank 1:  [====头2头3====]
                ↓ Hop1 (读 inner_block_len)           ↓ Hop1
CPU暂存0:  [头0头1]                              CPU暂存1:  [头2头3]
                ↓ Hop2 (offset=0)                     ↓ Hop2 (offset=1*inner_block_len)
                ↓                                     ↓                         ← D rank NPU →
                [头0头1 | 头2头3]  ← 拼装完整 block (4 heads)
```

---

## 3. MLA 和 DSv4 的切分方式

### MLA:KV cache 全复制,无需切分

MLA (Multi-head Latent Attention) 将 KV 压缩为 **单个 latent 向量**(kv_lora_rank=512),TP 间 **全复制而非按 head 切分**。每个 P rank 持有完整的 latent 数据,因此 D rank 只需从任意 1 个 P rank 拉取即可。

**切分数始终为 1**:

```python
# compute_tp_num_need_pulls (mooncake_d2rh_connector.py:437-448)
def compute_tp_num_need_pulls(num_key_value_heads, decode_tp_size,
                                prefill_tp_size, is_deepseek_mla):
    if is_deepseek_mla:
        return 1  # ★ MLA:每个 D rank 只需从 1 个 P rank 拉取(数据相同)
    num_d_block_heads = max(1, num_key_value_heads // decode_tp_size)
    num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
    return num_d_block_heads // num_p_block_heads
```

**随机选择 P rank**(因为所有 P rank 数据等价,随机选可实现负载均衡):

```python
# get_remote_ranks_for_req (mooncake_d2rh_connector.py:487-490)
if is_deepseek_mla or use_sparse:
    num_kv_head = 1  # ★ MLA:所有 P rank 数据等价,可随机选
```

**字节级**: `num_group_pulls=1` → `inner_block_len = block_len // 1 = block_len`(整块传输,不切分),`remote_tp_offset=0`(无偏移)。

### DSv4:与 MLA 相同的切分方式 + compress_ratio 调整

DSv4 (DeepSeek V4) 使用 DSA (DeepSeek Sparse Attention) + KV 压缩。检测标志是 `index_topk` 属性:

```python
# ascend_config.py:238-242
use_sparse = (
    vllm_config.model_config is not None
    and hasattr(vllm_config.model_config, "hf_text_config")
    and hasattr(vllm_config.model_config.hf_text_config, "index_topk")  # ★ DSv4 标志
)
```

**与 MLA 共用随机选 rank 路径**(同样 `num_kv_head=1`,`num_group_pulls=1`):

```python
# get_remote_ranks_for_req 中 (mooncake_d2rh_connector.py:462)
if prefill_tp_size > num_key_value_heads or is_deepseek_mla or use_sparse:
    # ★ DSv4 (use_sparse=True) 走随机分组路径,和 MLA 一样
    ...
```

**DSv4 的区别在于 compress_ratio 影响块大小和起始偏移**:

```python
# D2RHThread.__init__ 中提取 compress_ratio (mooncake_d2rh_connector.py:675-684)
self.group_compress_ratios: dict[int, int] = {}
for group_id, (group_spec, _) in self.kv_group2layeridx.items():
    compress_ratio = 1
    kv_cache_spec = group_spec.get("kv_cache_spec")
    if isinstance(kv_cache_spec, dict):
        for spec in kv_cache_spec.values():
            if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                compress_ratio = max(1, spec["compress_ratio"])
                break
    self.group_compress_ratios[group_id] = compress_ratio

# 传输时用 compress_ratio 计算起始偏移 (mooncake_d2rh_connector.py:897-900)
remote_kernel_block_size = self.block_size // remote_scale
remote_kernel_token_size = remote_kernel_block_size * self.group_compress_ratios[group_id]
# ★ 跳过已计算的 compressed block
remote_start_idx = req_meta.get("num_computed_tokens", 0) // remote_kernel_token_size
kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
```

`compress_ratio` 表示多少个 token 共享一个压缩后的 KV slot。例如 `compress_ratio=4` 时,4 个 token 的 KV 被压缩为 1 个 slot,因此每个 kernel block 覆盖 `block_size * 4` 个 token。传输时通过 `remote_start_idx` 跳过已计算的部分,减少传输量。

---

## 4. 示例汇总

### 普通 Attention (num_kv_heads=8):P TP=4,D TP=2

| 项目 | D rank 0 | D rank 1 |
|------|----------|----------|
| 拉取的 P rank | [0, 1] | [2, 3] |
| `num_group_pulls` | 2 | 2 |
| P rank 0/2 的 `remote_tp_offset` | 0 | 0 |
| P rank 1/3 的 `remote_tp_offset` | 1 | 1 |
| P rank 持有的 head | P0:[h0,h1] P1:[h2,h3] | P2:[h4,h5] P3:[h6,h7] |
| `inner_block_len` | `block_len // 2` | `block_len // 2` |
| Hop2 写入偏移 | offset=0: [h0,h1] offset=1: [h2,h3] | offset=0: [h4,h5] offset=1: [h6,h7] |
| D NPU block 最终 | [h0 h1 h2 h3] | [h4 h5 h6 h7] |

### MLA (DeepSeek V3):P TP=4,D TP=2

| 项目 | D rank 0 | D rank 1 |
|------|----------|----------|
| 拉取的 P rank | [2] (随机) | [0] (随机) |
| `num_group_pulls` | 1 | 1 |
| `remote_tp_offset` | 0 | 0 |
| `inner_block_len` | `block_len // 1 = block_len` (整块) | `block_len` |
| Hop2 写入偏移 | 0 | 0 |
| D NPU block 最终 | 完整 latent (与 P rank 2 相同) | 完整 latent (与 P rank 0 相同) |

### DSv4 (compress_ratio=4):P TP=4,D TP=2

| 项目 | D rank 0 | D rank 1 |
|------|----------|----------|
| 拉取的 P rank | [1] (随机) | [3] (随机) |
| `num_group_pulls` | 1 | 1 |
| `remote_tp_offset` | 0 | 0 |
| `inner_block_len` | `block_len // 1 = block_len` (整块) | `block_len` |
| `compress_ratio` | 4 | 4 |
| `remote_kernel_token_size` | `block_size * 4` | `block_size * 4` |
| `remote_start_idx` | `num_computed_tokens // (block_size*4)` | 同左 |
| D NPU block 最终 | 完整 KV (含 kv_lora + k_rope + indexer) | 同左 |
