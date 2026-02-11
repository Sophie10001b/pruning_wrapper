model_name=llama3.1-8b
model_path=/root/autodl-fs/modelscope_cache/llama3.1-8b

dynamic=static
style=unstructured
config_name=unstructured

benchmark_metric=ttft
num_repeat=20
sparsity=0.5

batch_size=(8)
seq_len=(1024)

python main.py \
    --model_name $model_name \
    --model_path $model_path \
    --dynamic $dynamic \
    --style $style \
    --config_name $config_name \
    --benchmark_metric $benchmark_metric \
    --num_repeat $num_repeat \
    --sparsity $sparsity \
    --batch_size ${batch_size[@]} \
    --seq_len ${seq_len[@]} \
    --cuda_graph \
    --liger_kernel
