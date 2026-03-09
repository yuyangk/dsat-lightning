"""Quick GPU / CUDA sanity-check script.

Run with:
    uv run python scripts/test_gpu.py
"""

import torch


def main() -> None:
    print("=" * 60)
    print("  GPU / CUDA Test")
    print("=" * 60)

    # ── PyTorch build info ───────────────────────────────────
    print(f"\nPyTorch version : {torch.__version__}")
    print(f"CUDA compiled   : {torch.version.cuda or 'N/A'}")
    print(f"cuDNN version   : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"cuDNN enabled   : {torch.backends.cudnn.enabled}")

    # ── Device detection ─────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available  : {cuda_available}")

    if not cuda_available:
        print("\n[FAIL] No CUDA-capable GPU detected. Exiting.")
        raise SystemExit(1)

    n_gpus = torch.cuda.device_count()
    print(f"GPU count       : {n_gpus}")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Memory           : {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute cap.     : {props.major}.{props.minor}")
        print(f"    Multi-processors : {props.multi_processor_count}")

    # ── Simple compute test ──────────────────────────────────
    device = torch.device("cuda")
    print(f"\nRunning compute test on {torch.cuda.get_device_name(0)} ...")

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    # Warm-up
    _ = a @ b
    torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    tflops = 2 * 4096**3 / (elapsed_ms / 1000) / 1e12
    print(f"  MatMul 4096x4096 : {elapsed_ms:.2f} ms  ({tflops:.2f} TFLOPS)")

    # ── Memory summary ───────────────────────────────────────
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"\n  Memory allocated  : {alloc:.1f} MB")
    print(f"  Memory reserved   : {reserved:.1f} MB")

    # ── Cleanup ──────────────────────────────────────────────
    del a, b, c
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  [PASS] GPU is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
