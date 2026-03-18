print_autotune=1
nsys_profile=1
model_name=llama3.1-8b
model_path=/home/hk-project-p0022189/tum_yvc3016/anhao/haozhe/modelscope_cache/models/LLM-Research/Meta-Llama-3.1-8B

dynamic=static
style=dense
config_name=dense

benchmark_metric=tpot
num_repeat=3
sparsity=0.5

batch_size=(1)
seq_len=(32768)

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
        --sparsity $sparsity \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --cuda_graph \
        --liger_kernel

else
    nsys_prefix="${benchmark_metric}_${batch_size[0]}_${seq_len[0]}_${dynamic}_${style}_${config_name}_${sparsity}"
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
