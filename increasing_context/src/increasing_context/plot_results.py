import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def format_tokens(val):
    if val >= 1024:
        return f"{int(val // 1024)}k"
    return str(int(val))

def load_data(filepath="results.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def parse_oom_memory(error_str):
    """Extract 'X.XX GiB memory in use' from a CUDA OOM error message."""
    if not error_str:
        return None
    match = re.search(r"Process \d+ has ([\d.]+) GiB memory in use", error_str)
    return float(match.group(1)) if match else None

def main():
    data = load_data("results.json")

    # Group by model
    models = {}
    for entry in data:
        m = entry["model"]
        if m not in models:
            models[m] = {"ok": [], "oom": []}
        if entry["status"] == "ok":
            models[m]["ok"].append(entry)
        else:
            models[m]["oom"].append(entry)

    # Style configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15
    })

    display_names = {
        "Qwen2.5-1.5B": "Qwen2.5-1.5B",
        "Qwen3.5-0.8B": "Qwen3.5-0.8B (Standard)",
        "Qwen3.5-0.8B-optimized": "Qwen3.5-0.8B (Optimized)",
    }
    colors = {
        "Qwen2.5-1.5B": "#1f77b4",            # blue
        "Qwen3.5-0.8B": "#ff7f0e",            # orange
        "Qwen3.5-0.8B-optimized": "#2ca02c",  # green
    }
    markers = {
        "Qwen2.5-1.5B": "o",
        "Qwen3.5-0.8B": "s",
        "Qwen3.5-0.8B-optimized": "^",
    }
    oom_text_offsets = {
        "Qwen3.5-0.8B": (0.22, 18.0),
        "Qwen3.5-0.8B-optimized": (0.22, 13.5),
        "Qwen2.5-1.5B": (0.22, 18.0),
    }

    # Real usable VRAM ceiling, derived from the CUDA OOM messages themselves
    # ("GPU 0 has a total capacity of 22.06 GiB..."), NOT the 24 GB nameplate figure.
    GPU_USABLE_VRAM = 22.06
    GPU_NAMEPLATE_VRAM = 24.0

    # ==========================================
    # Plot 1: Context Length vs Peak VRAM (Single)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    for m_name, runs in models.items():
        ok_runs = sorted(runs["ok"], key=lambda x: x["context_length"])
        ctx = [r["context_length"] for r in ok_runs]
        vram = [r["peak_memory_gb"] for r in ok_runs]

        color = colors.get(m_name, "#333333")
        marker = markers.get(m_name, "o")
        label = display_names.get(m_name, m_name)

        ax.plot(ctx, vram, marker=marker, linewidth=2.2, markersize=7, label=label, color=color)

        # Plot OOM markers, using the real "memory in use at failure" value
        # parsed from the error string when available, rather than a shared constant.
        for oom in sorted(runs["oom"], key=lambda x: x["context_length"]):
            oom_ctx = oom["context_length"]
            mem_at_failure = parse_oom_memory(oom.get("error", ""))
            y_val = mem_at_failure if mem_at_failure is not None else GPU_USABLE_VRAM

            # Dashed connector from the last successful point to the OOM marker,
            # visually distinguishing "crash snapshot" from a converged peak reading.
            if ok_runs:
                last_ok = ok_runs[-1]
                ax.plot(
                    [last_ok["context_length"], oom_ctx],
                    [last_ok["peak_memory_gb"], y_val],
                    linestyle=":", linewidth=1.5, color=color, alpha=0.6
                )

            ax.scatter(oom_ctx, y_val, color="red", marker="x", s=120, linewidth=3, zorder=5)
            x_factor, y_pos = oom_text_offsets.get(m_name, (0.22, 23.0))
            ax.annotate(
                f"{label}\nOOM at {format_tokens(oom_ctx)}\n"
                f"(mem in use at failure: {y_val:.1f} GB)",
                xy=(oom_ctx, y_val),
                xytext=(oom_ctx * x_factor, y_pos),
                ha="right",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=8.5,
                color="darkred",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffeeee", ec="red", lw=1)
            )

    # Real usable VRAM ceiling (honest limit, not nameplate)
    ax.axhline(GPU_USABLE_VRAM, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"A10G Usable VRAM (~{GPU_USABLE_VRAM:.1f} GB, from OOM logs)")
    ax.text(8192, GPU_USABLE_VRAM + 0.3,
            f"Usable Capacity: ~{GPU_USABLE_VRAM:.1f} GB (nameplate {GPU_NAMEPLATE_VRAM:.0f} GB)",
            color="#d62728", fontsize=9.0, fontweight="bold", zorder=10)

    ax.set_xscale("log", base=2)
    all_ctx = [128, 512, 2048, 8192, 32768, 65536, 131072, 262144]
    ax.set_xticks(all_ctx)
    ax.set_xticklabels([format_tokens(x) for x in all_ctx])

    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Peak VRAM Usage (GB)")
    ax.set_title("Context Length vs Peak VRAM (A10G GPU)")
    ax.set_ylim(0, 26)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig("ctx_vs_vram.png", dpi=300)
    if os.path.isdir("plots"):
        plt.savefig(os.path.join("plots", "ctx_vs_vram.png"), dpi=300)
        print("Saved plots/ctx_vs_vram.png")
    plt.close()
    print("Saved ctx_vs_vram.png")

if __name__ == "__main__":
    main()
