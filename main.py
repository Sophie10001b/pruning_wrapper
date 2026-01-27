import argparse

from transformers import set_seed

from prologue import init_model
from benchmark.profiler import ModelProfiler

def parse_args():
    parser = argparse.ArgumentParser(description="Pruning Wrapper")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dynamic", type=str, default="token_dynamic", help="Dynamic type")
    parser.add_argument("--style", type=str, default="skipgpt", help="Wrapper style")

    parser.add_argument("--benchmark_metric", type=str, default="ttft", help="Benchmark metric")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num_repeat", type=int, default=500, help="Number of repeat iterations")
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA graph")
    parser.add_argument("--liger_kernel", action="store_true", help="Enable Liger Kernel")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.liger_kernel:
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen3, apply_liger_kernel_to_qwen2
        liger_kernel_kwargs = {'rope': False, 'rms_norm': True, 'swiglu': False}
        apply_liger_kernel_to_llama(**liger_kernel_kwargs)
        apply_liger_kernel_to_qwen2(**liger_kernel_kwargs)
        apply_liger_kernel_to_qwen3(**liger_kernel_kwargs)

    model, tokenizer = init_model(args)
    profiler = ModelProfiler(
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    if args.benchmark_metric == "ttft":
        profile_res = profiler.profile_ttft(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.num_warmup,
            repeat=args.num_repeat,
            cuda_graph=args.cuda_graph,
        )
    elif args.benchmark_metric == "tpot":
        profile_res = profiler.profile_tpot(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.num_warmup,
            repeat=args.num_repeat,
            cuda_graph=args.cuda_graph,
        )

if __name__ == "__main__":
    main()