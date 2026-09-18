# Windows gfx1102 CPU-MoE bundle (for Maxritz/FreeToken-ROCm)

This branch is the Windows/ROCm serve stack that ran `openai/gpt-oss-20b` at
~10.5 tok/s on a Framework Laptop 16 (RX 7700S, gfx1102, 8 GB). It is meant
for [Maxritz/FreeToken-ROCm](https://github.com/Maxritz/FreeToken-ROCm), not
as a dump of FlashML PR #132.

Companion tvm-ffi overlay (hipcc JIT):
https://github.com/AronVanAmmers/tvm-ffi/tree/windows-hip-0.1.13-post3

## Hardware

- Machine: Framework Laptop 16
- GPU: Radeon RX 7700S, gfx1102, 8 GB (pin HIP_VISIBLE_DEVICES=0; 780M is device 1)
- CPU: Ryzen 9 7940HS, 8 cores / 16 threads, AVX-512-BF16
- RAM: 96 GB installed, 8.2 GB hardware reserved (~88 GB visible to Windows)
- OS: Windows 11
- HIP: TheRock nightly gfx110X-all (ROCm 7.14.0a20260612)
- Torch: 2.11.0+rocm7.14.0a20260612
- Model: openai/gpt-oss-20b HF MXFP4, 12.84 GB

## Serve command (the 10.5 tok/s path)

8 GB cannot fused-hold this checkpoint. Live PCIe expert fetch was ~0.02 tok/s.
CPU experts with fetch disabled is what served:

```powershell
$env:HIP_VISIBLE_DEVICES = "0"
$env:TVM_FFI_ROCM_ARCH_LIST = "gfx1102"
python -m freetoken.cli serve --model C:\models\gpt-oss-20b `
  --moe-backend hybrid --moe-cpu-threads 8 --moe-hybrid-max-fetch 0 `
  --moe-cache-auto --host 127.0.0.1 --port 1919
```

`--moe-hybrid-max-fetch 0` is a bypass, not a HIP gather fix.

Thread: https://github.com/FlashML-org/FreeToken/issues/82
