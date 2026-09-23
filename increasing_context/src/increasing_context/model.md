# Model Configurations & KV Cache Calculations

## 1. Qwen 2.5 (1.5B)

### KV Cache Calculation

The KV cache size is given by

$$
\text{KV Cache Size} = 2 \times N_{\text{layers}} \times N_{\text{kv}} \times d_{\text{head}} \times L \times B
$$

where:

* $N_{\text{layers}} = 28$ (`num_hidden_layers` in config)
* $N_{\text{kv}} = 2$ (`num_key_value_heads` in config — GQA with a $6{:}1$ query-to-KV ratio; $N_{\text{attn}} = 12$)
* $d_{\text{head}} = \frac{d_{\text{model}}}{N_{\text{attn}}} = \frac{1536}{12} = 128$ (`head_dim`)
* $B = 2\ \text{bytes}$ (element size for `bfloat16` / `float16`)
* $L = \text{sequence length}$ (number of tokens in the sequence)

Therefore, the KV cache size per token is

$$
\text{KV Cache per Token} = 2 \times 28 \times 2 \times 128 \times 2 = 28{,}672\ \text{bytes} \approx 28\ \text{KiB/token}
$$

Thus, for a sequence of length $L$:

$$
\text{Total KV Cache} = 28{,}672 \times L\ \text{bytes}
$$

> **Result:** `Total KV Cache = 28,672 × L bytes`

### Config (`Qwen2Config`)

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 1000000.0,
    "rope_type": "default"
  },
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "5.17.0",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

---

## 2. Qwen 3.5 (0.8B)

### KV Cache Calculation

Qwen 3.5 uses a **hybrid architecture** with $24$ layers in total:

* $6$ full-attention layers (every fourth layer; `full_attention_interval = 4`) that store a dynamic KV cache.
* $18$ linear-attention layers (Gated DeltaNet) that maintain a fixed-size recurrent state rather than a token-by-token KV cache.

Therefore, the dynamic KV cache is

$$
\text{KV Cache Size} = 2 \times N_{\text{attn layers}} \times N_{\text{kv}} \times d_{\text{head}} \times L \times B
$$

where:

* $N_{\text{attn layers}} = 6$ (number of `full_attention` layers)
* $N_{\text{kv}} = 2$ (`num_key_value_heads`; $N_{\text{attn}} = 8$)
* $d_{\text{head}} = 256$ (`head_dim`)
* $B = 2\ \text{bytes}$ (element size for `bfloat16` / `float16`)
* $L = \text{sequence length}$ (number of tokens in the sequence)

The KV cache size per token is therefore

$$
\text{KV Cache per Token} = 2 \times 6 \times 2 \times 256 \times 2 = 12{,}288\ \text{bytes} \approx 12\ \text{KiB/token}
$$

Thus, for a sequence of length $L$:

$$
\text{Total KV Cache} = 12{,}288 \times L\ \text{bytes}
$$

> **Result:** `Total KV Cache = 12,288 × L bytes`

### Config (`Qwen3_5Config`)

```json
{
  "architectures": [
    "Qwen3_5ForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "image_token_id": 248056,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "bos_token_id": null,
    "dtype": "float16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3584,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "pad_token_id": null,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [
        11,
        11,
        10
      ],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    },
    "tie_word_embeddings": true,
    "use_cache": true,
    "vocab_size": 248320
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.17.0",
  "video_token_id": 248057,
  "vision_config": {
    "deepstack_visual_indexes": [],
    "depth": 12,
    "dtype": "float16",
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 768,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "model_type": "qwen3_5_vision",
    "num_heads": 12,
    "num_position_embeddings": 2304,
    "out_hidden_size": 1024,
    "patch_size": 16,
    "rope_parameters": {
      "rope_theta": 10000.0,
      "rope_type": "axial"
    },
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 248054,
  "vision_start_token_id": 248053
}
```

### Runtime Notes

> [!NOTE]
> The unoptimized fallback implementation of linear attention / DeltaNet rules, used when `causal_conv1d` and `flash-linear-attention` are unavailable, materializes large intermediate activation tensors during long-context runs. This can contribute to higher peak memory usage.

```text
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
[transformers] `causal_conv1d_update` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `fused_recurrent_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
```

## 3. Fixing the above import issues - Qwen3.5-optimized
To install the packages mentioned above. I had to change the model image to a CUDA image, which had NVCC. This was required to compile causal conv1, which took a lot of time. I don't know how much improvement it provides to our model but still it has been compiled and used. 

```python
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
```
The reduction in VRAM usage can be seen in [README](../../README.md)
<!-- TODO: Check how much improvement does each of this optimisation bring in and learn about it -->
