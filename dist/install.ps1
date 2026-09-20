# ============================================================
#  FreeToken for Windows + AMD GPUs - one-command installer
#
#  What this does (all automatic):
#    1. makes a private Python environment (.venv) so nothing
#       else on your PC gets touched
#    2. installs the AMD GPU torch/ROCm wheels you downloaded
#    3. installs the engine + its small helper packages
#    4. patches 3 upstream bugs (automatic, safe to re-run)
#
#  Run it from inside the cloned repo folder:
#    powershell -ExecutionPolicy Bypass -File dist\install.ps1
#
#  Need help? Check PORT_REQUIREMENTS.md first.
# ============================================================
param(
    [string]$Py = "py -3.12",         # leave as-is if you installed Python 3.12 normally
    [string]$Arch = "",               # your GPU family (gfx1200 = RX 9060 XT, gfx1201 = RX 9070 XT); auto-detected if empty
    [string]$Stamp = "",              # pin every AMD wheel to one nightly stamp, e.g. 10.1.0a20260806 (empty = latest)
    [string]$TorchVer = "2.11.0",     # torch build to pair with the stamp (pyproject pins >=2.11,<2.12)
    [string]$WheelDir = ""            # folder holding AMD .whl files (see below)
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"

# ---- GPU family auto-detection (name -> gfx arch), overridable via -Arch ----
if (-not $Arch) {
    $gpuName = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" } |
                Select-Object -First 1 -ExpandProperty Name)
    $Arch = switch -Regex ($gpuName) {
        "RX 9070"           { "gfx1201"; break }
        "RX 9060"           { "gfx1200"; break }
        "RX 7900"           { "gfx1100"; break }
        "RX 77\d0|RX 7800"  { "gfx1101"; break }
        "RX 76\d0"          { "gfx1102"; break }
        default             { "" }
    }
    if (-not $Arch) { throw "Could not map GPU '$gpuName' to a gfx arch - pass -Arch (e.g. -Arch gfx1200)" }
    Write-Host "      detected GPU: $gpuName -> $Arch" -ForegroundColor Gray
}
$tag = ($Py -replace '\s', '') + "-" + $Arch.Replace(',', '+')
if (-not $WheelDir) { $WheelDir = Join-Path $REPO "rocm-wheels\$tag" }
$VENV = Join-Path $REPO ".venv"
$PYEXE = "$VENV\Scripts\python.exe"

Write-Host ""
Write-Host "  FreeToken installer for Windows + AMD" -ForegroundColor Cyan
Write-Host "  --------------------------------------"

# ---- Step 1: private python environment -------------------------------
Write-Host "`n[1/5] Creating a private Python environment (.venv) ..." -ForegroundColor Yellow
Invoke-Expression "$Py -m venv `"$VENV`""
# A fresh venv ships pip but no setuptools on 3.12+, and step 2 installs the `rocm`
# metapackage sdist with --no-build-isolation -- without setuptools that dies with
# "Cannot import 'setuptools.build_meta'", leaving torch unable to import rocm_sdk.
& $PYEXE -m pip install --upgrade pip setuptools wheel

# ---- Step 2: AMD GPU wheels -------------------------------------------
Write-Host "[2/5] AMD GPU wheels (torch / ROCm) ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
# reuse a device wheel dropped into the repo root (skips the biggest download)
Get-ChildItem $REPO -Filter "rocm_sdk_device_$Arch-*.whl" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName $WheelDir -ErrorAction SilentlyContinue }
if (-not (Get-ChildItem "$WheelDir" -Recurse -Filter "torch-*.whl" -ErrorAction SilentlyContinue)) {
    Write-Host "      downloading AMD wheels (~2 GB, one time only; already-present files are skipped) ..." -ForegroundColor Gray
    # every AMD wheel must share ONE nightly stamp (see PORT_REQUIREMENTS.md #3)
    $rocmSpec  = "rocm[libraries,devel,device-$Arch]"
    $torchSpec = "torch"; $devSpec = "amd-torch-device-$Arch"
    if ($Stamp) {
        $rocmSpec  = "rocm[libraries,devel,device-$Arch]==$Stamp"
        $torchSpec = "torch==$TorchVer+rocm$Stamp"
        $devSpec   = "amd-torch-device-$Arch==$TorchVer+rocm$Stamp"
    }
    Invoke-Expression "$Py -m pip download --index-url $INDEX -d `"$WheelDir`" `"$rocmSpec`""
    Invoke-Expression "$Py -m pip download --no-deps --index-url $INDEX -d `"$WheelDir`" `"$torchSpec`" `"$devSpec`""
}
& $PYEXE -m pip install (Get-ChildItem $WheelDir -Recurse -Filter *.whl | ForEach-Object { $_.FullName }) --no-deps --force-reinstall
# the 'rocm' metapackage sdist provides the rocm_sdk module torch's _rocm_init imports
$rocmSdist = Get-ChildItem $WheelDir -Recurse -Filter "rocm-*.tar.gz" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($rocmSdist) { & $PYEXE -m pip install $rocmSdist.FullName --no-deps --no-build-isolation }

# ---- Step 3: engine + helpers -----------------------------------------
# freetoken itself is installed --no-deps, so every runtime dep from pyproject.toml
# must be listed here (CUDA-only extras excluded: flashinfer, sglang-kernel).
Write-Host "[3/5] Installing FreeToken + helpers ..." -ForegroundColor Yellow
& $PYEXE -m pip install "triton-windows>=3.7.1" apache-tvm-ffi==0.1.13.post3 msgpack pyzmq psutil requests aiohttp partial_json_parser gguf `
    einops fastapi uvicorn pydantic openai prompt_toolkit "transformers>=5.5,<6" huggingface_hub safetensors `
    "numpy>=2.0,<2.5" tqdm modelscope tornado ninja setuptools wheel numba
# flashlib (MoE expert-cache slot_cache kernel) is a real runtime dep, but it declares
# torch>=2.0 -- resolving that would pull the CUDA torch from PyPI over the ROCm wheel
# installed in step 2. Its other deps (triton-windows, numpy, numba, tqdm) are above.
& $PYEXE -m pip install flashlib==0.3.0 --no-deps
    "numpy>=2.0,<2.5" tqdm modelscope tornado ninja numba setuptools wheel
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
& $PYEXE -m pip install -e "$REPO" --no-deps --no-build-isolation
Remove-Item Env:FREETOKEN_SKIP_CUDA_EXT

# ---- Step 4: upstream patches -----------------------------------------
Write-Host "[4/5] Applying 3 small compatibility patches ..." -ForegroundColor Yellow
& $PYEXE "$REPO\dist\patch_upstream.py"

# ---- Step 5: verify -----------------------------------------------------
Write-Host "[5/5] Checking your GPU ..." -ForegroundColor Yellow
& $PYEXE -c "import torch; print('      torch', torch.__version__, '| HIP', torch.version.hip); print('      GPU:', torch.cuda.get_device_name(0)); arch=torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]; print('      arch:', arch); assert arch=='$Arch', f'GPU arch {arch} != installed device wheels ($Arch) - rerun with -Arch {arch}'"
# $ErrorActionPreference does not apply to native exit codes, so check it explicitly --
# otherwise a failed GPU check still prints "All done!" over a broken install.
if ($LASTEXITCODE -ne 0) { throw "GPU check failed - see the traceback above; the install is not usable." }

Write-Host ""
Write-Host "  All done! To chat with a model:" -ForegroundColor Green
Write-Host "    powershell -File dist\run-server.ps1 -Model <path-to-your-model>" -ForegroundColor White
Write-Host "  then open http://localhost:1420 in your browser.`n" -ForegroundColor Green
