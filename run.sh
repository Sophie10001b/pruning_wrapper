print_autotune=1
nsys_profile=0
model_name=llama3.1-8b
model_path=...

benchmark_metric=ttft
num_repeat=10

batch_size=(1)
seq_len=(32768)
sparsity=(0.5)

dynamic=static
style=dense
config_name=dense

if [[ $nsys_profile == 0 ]]; then
    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --cuda_graph

else
    nsys_prefix="${benchmark_metric}_${batch_size[0]}_${seq_len[0]}_${dynamic}_${style}_${config_name}_${sparsity[0]}"
    nsys profile --trace=cuda,nvtx -o ${nsys_prefix}.nsys-rep --force-overwrite true python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity $sparsity \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel
fi
