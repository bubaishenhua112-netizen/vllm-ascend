export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_PROCESS_LOG_PATH=/home/zjs/logs_pd
export ASCEND_GLOBAL_LOG_LEVEL=1

export ASCEND_HOST_LOG_FILE_NUM=1000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=1

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'

rm -rf /home/zjs/logs_pd

nic_name="eth0"
local_ip="141.61.73.112"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

vllm serve /mnt/share/weight/Qwen3-30B-A3B-Instruct-2507/ \
--served-model-name ds \
--trust-remote-code \
--max-num-seqs 24 \
--tensor-parallel-size 1 \
--data-parallel-size 1 \
--data-parallel-size-local 1 \
--port 30099 \
--max_model_len 30000 \
--max-num-batched-tokens 256 \
--enforce-eager \
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--async-scheduling \
--additional-config='{"enable_cpu_binding":false}' \
--kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "20400", "engine_id": "2",
     "kv_connector_extra_config": {
         "prefill": {
             "dp_size": 1,
             "tp_size": 1
         },
         "decode": {
             "dp_size": 1,
             "tp_size": 1
         }
     }
 }'
