import modal
import time
import json

app = modal.App("kv-cache-optimized-kernels")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang-14", "lld")
    .env({"CC": "clang-14", "CXX": "clang++-14"})
    .uv_pip_install(
        "torch",
        "torchvision",
        "accelerate",
        "transformers>=4.57.0",
        "pillow",
        "ninja",
        "packaging",
    )
    # Compiled CUDA extensions - now have nvcc available via the devel base image,
    # unlike debian_slim() which only ships the CUDA runtime, not the compiler toolchain.
    .run_commands(
        "pip install causal-conv1d --no-build-isolation",
    )
    .run_commands(
        "pip install flash-linear-attention",
    )
)

CONTEXT_LENGTHS_QWEN35 = [128, 512, 2048, 8192, 32768, 65536, 131072, 262144]
MAX_CTX = max(CONTEXT_LENGTHS_QWEN35)


def build_prompt_of_length(processor, target_len):
    filler = "The quick brown fox jumps over the lazy dog. " * (MAX_CTX // 8 + 1000)
    ids = processor.tokenizer(filler, return_tensors="pt")["input_ids"][0]
    if ids.shape[0] < target_len:
        raise ValueError(
            f"Filler too short for {target_len} tokens, got {ids.shape[0]}"
        )
    ids = ids[:target_len]
    return ids.unsqueeze(0)


@app.function(
    gpu="A10G",
    image=image,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_qwen35_optimized():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    # sanity check: confirm the optimized kernels actually imported successfully
    try:
        import causal_conv1d

        causal_conv1d_ok = True
    except ImportError as e:
        causal_conv1d_ok = False
        print(f"causal_conv1d import failed: {e}")

    try:
        import fla  # flash-linear-attention

        fla_ok = True
    except ImportError as e:
        fla_ok = False
        print(f"flash-linear-attention import failed: {e}")

    print(f"causal_conv1d available: {causal_conv1d_ok}")
    print(f"flash-linear-attention available: {fla_ok}")

    model_id = "Qwen/Qwen3.5-0.8B"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    results = []
    for ctx_len in CONTEXT_LENGTHS_QWEN35:
        try:
            input_ids = build_prompt_of_length(processor, ctx_len).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start = time.time()
            with torch.no_grad():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                )
            torch.cuda.synchronize()
            latency = time.time() - start

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

            entry = {
                "model": "Qwen3.5-0.8B-optimized",
                "context_length": ctx_len,
                "peak_memory_gb": round(peak_mem_gb, 4),
                "latency_s": round(latency, 4),
                "status": "ok",
                "causal_conv1d_available": causal_conv1d_ok,
                "flash_linear_attention_available": fla_ok,
            }
        except Exception as e:
            entry = {
                "model": "Qwen3.5-0.8B-optimized",
                "context_length": ctx_len,
                "status": "error",
                "error": str(e)[:300],
            }
        print(entry)
        results.append(entry)

    return results


@app.local_entrypoint()
def main():
    print("Benchmarking Qwen3.5-0.8B with optimized kernels...")
    results = benchmark_qwen35_optimized.remote()

    with open("results_optimized.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll results:")
    for r in results:
        print(r)
