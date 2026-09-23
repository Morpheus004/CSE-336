import modal
import time
import json

app = modal.App("kv-cache-comparison")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "accelerate", "transformers>=4.57.0", "pillow"
)

CONTEXT_LENGTHS_QWEN25 = [
    128,
    512,
    2048,
    8192,
    32768,
    65536,
    131072,
]  # matches confirmed max_position_embeddings
CONTEXT_LENGTHS_QWEN35 = [
    128,
    512,
    2048,
    8192,
    32768,
    65536,
    131072,
    262144,
]  # confirmed native max 262144


MAX_CTX = max(max(CONTEXT_LENGTHS_QWEN25), max(CONTEXT_LENGTHS_QWEN35))


def build_prompt_of_length(tok, target_len, is_processor=False):
    # ~9 tokens per repeat of filler; pad with large safety margin for largest context length
    filler = "The quick brown fox jumps over the lazy dog. " * (MAX_CTX // 8 + 1000)
    if is_processor:
        ids = tok.tokenizer(filler, return_tensors="pt")["input_ids"][0]
    else:
        ids = tok(filler, return_tensors="pt")["input_ids"][0]
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
def benchmark_qwen25():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    results = []
    for ctx_len in CONTEXT_LENGTHS_QWEN25:
        try:
            input_ids = build_prompt_of_length(tokenizer, ctx_len).to("cuda")
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
                "model": "Qwen2.5-1.5B",
                "context_length": ctx_len,
                "peak_memory_gb": round(peak_mem_gb, 4),
                "latency_s": round(latency, 4),
                "status": "ok",
            }
        except Exception as e:
            entry = {
                "model": "Qwen2.5-1.5B",
                "context_length": ctx_len,
                "status": "error",
                "error": str(e)[:300],
            }
        print(entry)
        results.append(entry)

    return results


@app.function(
    gpu="A10G",
    image=image,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_qwen35():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    model_id = "Qwen/Qwen3.5-0.8B"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    results = []
    for ctx_len in CONTEXT_LENGTHS_QWEN35:
        try:
            input_ids = build_prompt_of_length(
                processor, ctx_len, is_processor=True
            ).to("cuda")
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
                "model": "Qwen3.5-0.8B",
                "context_length": ctx_len,
                "peak_memory_gb": round(peak_mem_gb, 4),
                "latency_s": round(latency, 4),
                "status": "ok",
            }
        except Exception as e:
            entry = {
                "model": "Qwen3.5-0.8B",
                "context_length": ctx_len,
                "status": "error",
                "error": str(e)[:300],
            }
        print(entry)
        results.append(entry)

    return results


@app.local_entrypoint()
def main():
    print("Benchmarking Qwen2.5-1.5B...")
    r1 = benchmark_qwen25.remote()
    with open("results.json", "w") as f:
        json.dump(r1, f, indent=2)
    print("Saved partial results after Qwen2.5-1.5B.")

    print("Benchmarking Qwen3.5-0.8B...")
    r2 = benchmark_qwen35.remote()

    all_results = r1 + r2
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results:")
    for r in all_results:
        print(r)
