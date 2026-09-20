# ============================================================
#  FreeToken server launcher (Windows + AMD)
#
#  Usage:
#    powershell -File dist\run-server.ps1 -Model <path-to-model>
#
#  Optional switches:
#    -Arch gfx1201          your GPU family
#    -RocmPath <folder>     where the AMD ROCm runtime lives
#                           (not needed if HIP_PATH is already set)
#    -Port 1919             API port
#    -KVPages 4096          cap KV cache (needed for big dense models)
#    -ExtraArgs "..."       anything else for `ft serve`, as ONE space-separated
#                           string:  -ExtraArgs "--moe-backend offload --num-pages 4096"
#                           (`powershell -File` does no PowerShell parsing, so the
#                           array form "--a","b" arrives as the literal string
#                           "--a,b" -- both are split apart below)
#
#  Pass --cuda-graph-max-bs 0 in -ExtraArgs. On gfx1201 CUDA graphs are a large
#  LOSS, not a win: measured on Gemma-4-26B-A4B QAT q4_0 with identical settings,
#  eager decodes at 37.9 tok/s and graph replay at 10.4.
#
#  When it says READY, open http://localhost:1420 and chat.
# ============================================================
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Arch = "gfx1201",
    [string]$RocmPath = "",
    [int]$Port = 1919,
    [int]$KVPages = 0,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$LogDir = "$env:TEMP\freetoken-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
# Clear last run's logs up front: the redirect below only truncates them once cmd
# reaches the `ft serve` line (after vcvarsall), and until then the readiness poll
# is reading the PREVIOUS run - a stale traceback aborts this launch instantly.
Remove-Item "$LogDir\serve.log", "$LogDir\serve_err.log" -ErrorAction SilentlyContinue
if (Test-Path "$LogDir\serve.log") {
    throw "$LogDir\serve.log is still locked by a previous run, so this launch would read the old log and report its error as this one's. A scheduler worker outlived a failed server and still holds the inherited handle: it is a python.exe running ``multiprocessing.spawn``, NOT anything matching ``freetoken``. Run dist\stop-server.ps1 - it sweeps those - then retry."
"$LogDir\serve.log", "$LogDir\serve_err.log" | ForEach-Object {
    if (Test-Path $_) {
        try {
            Remove-Item $_ -Force -ErrorAction Stop
        } catch {
            # The desktop app may tail the log with delete sharing disabled.
            Clear-Content $_ -Force -ErrorAction Stop
        }
    }
}

if (-not $RocmPath) {
    if ($env:HIP_PATH) { $RocmPath = $env:HIP_PATH }
    else {
        # The ROCm runtime ships INSIDE the venv install.ps1 built -- the TheRock wheels
        # unpack it to site-packages\_rocm_sdk_core (lib\llvm\..., the layout the env vars
        # below assume). Ask Python where that is before telling the user to go find a
        # system ROCm install they never had to make; HIP_PATH is set by nothing here.
        $pyExe = Join-Path $REPO ".venv\Scripts\python.exe"
        if (-not (Test-Path $pyExe)) { $pyExe = "python" }
        $found = & $pyExe -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))" 2>$null
        if ($found) { $found = ([string]$found).Trim() }
        if ($found -and (Test-Path (Join-Path $found "lib\llvm\bin\clang.exe"))) {
            $RocmPath = $found
            Write-Host "ROCm runtime: $RocmPath (from the venv)" -ForegroundColor DarkGray
        }
    }
    if (-not $RocmPath) { throw "Where is the AMD ROCm runtime? It is not in this repo's .venv and HIP_PATH is unset. Pass -RocmPath, or re-run dist\install.ps1 to build the venv." }
}
# `powershell -File` hands every argument through as a literal string -- there is no
# PowerShell parser on that path -- so -ExtraArgs "--moe-backend","offload" arrives as
# the single element "--moe-backend,offload", which `ft serve` rejects as one
# unrecognized argument. Split every element on commas and whitespace so the array form
# (dot-sourced, or -Command) and the string form (-File) both reach ft the same way.
$ExtraArgs = @($ExtraArgs | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
if ($KVPages -gt 0) { $ExtraArgs += "--num-pages", "$KVPages" }

# The generated runner does `cd /d %TEMP%` before calling ft, so a relative -Model
# (the form the README and every note use: modelsoo.gguf) resolves against %TEMP%,
# misses, and transformers then treats it as a HUGGING FACE REPO ID -- the error names
# repo-id character rules and never mentions the path. Resolve it here, against the
# caller's cwd and then the repo, before it can turn into a download attempt.
if (-not [System.IO.Path]::IsPathRooted($Model)) {
    $candidate = Join-Path (Get-Location) $Model
    if (-not (Test-Path $candidate)) { $candidate = Join-Path $REPO $Model }
    if (-not (Test-Path $candidate)) {
        throw "Model not found: '$Model' (looked in $(Get-Location) and $REPO). Pass a full path."
    }
    $Model = (Resolve-Path $candidate).Path
}

# engine binary: prefer the repo venv this installer created, fall back to PATH
$ft = Join-Path $REPO ".venv\Scripts\ft.exe"
if (-not (Test-Path $ft)) { $ft = "ft" }

# vcvarsall lives under either Program Files root (Build Tools installs land in the
# x86 one), so ask vswhere first and only then fall back to scanning both roots.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = $null
if (Test-Path $vswhere) {
    $vsRoot = & $vswhere -latest -products * -find "VC\Auxiliary\Build\vcvarsall.bat" 2>$null | Select-Object -First 1
    if ($vsRoot) { $vcvars = $vsRoot }
}
if (-not $vcvars) {
    $vcvars = Get-ChildItem "$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio" `
                  -Recurse -Filter vcvarsall.bat -ErrorAction SilentlyContinue |
              Select-Object -First 1 -ExpandProperty FullName
}
if (-not $vcvars) { Write-Warning "vcvarsall.bat not found - JIT DLL links may fail to find the MSVC CRT." }
$vcvars = @("${env:ProgramFiles(x86)}\Microsoft Visual Studio", "$env:ProgramFiles\Microsoft Visual Studio") |
          Where-Object { Test-Path $_ } |
          ForEach-Object { Get-ChildItem $_ -Recurse -Filter vcvarsall.bat -ErrorAction SilentlyContinue } |
          Select-Object -First 1 -ExpandProperty FullName
if (-not $vcvars) {
    throw "Visual Studio Build Tools with the C++ workload and Windows SDK are required (vcvarsall.bat not found)."
}

$cmd = @"
call `"$vcvars`" x64 >nul
set HIP_PATH=$RocmPath
rem tvm_ffi's JIT resolves the toolchain from ROCM_HOME (not HIP_PATH) and dies at
rem the first kernel build without it -- long after the model has loaded.
rem ROCM_HOME only: setting ROCM_PATH too makes clang look for the device bitcode
rem at %ROCM_PATH%\amdgcn\bitcode, which is not the TheRock wheel layout (it lives
rem under lib\llvm\), and every kernel build then fails to find it.
set ROCM_HOME=$RocmPath
set TVM_FFI_ROCM_ARCH_LIST=$Arch
rem Without this torch's JIT compiles every extension for EVERY visible card --
rem including the CPU's iGPU (gfx1036 on a 9800X3D), which is not the serving
rem device and whose build failure kills the backend worker after a full load.
set PYTORCH_ROCM_ARCH=$Arch
set TRITON_OVERRIDE_ARCH=$Arch
set ROCM_SDK_TARGET_FAMILY=$Arch
set PYTORCH_ROCM_ARCH=$Arch
set HIP_DEVICE_LIB_PATH=$RocmPath\lib\llvm\amdgcn\bitcode
set ROCM_HOME=$RocmPath
set ROCM_PATH=$RocmPath
set TVM_FFI_CACHE_DIR=$REPO\.tvm-ffi-cache
set PATH=$RocmPath\bin;%PATH%
set "CC=$RocmPath\lib\llvm\bin\clang-cl.exe"
cd /d %TEMP%
"$ft" serve --model "$Model" --port $Port $($ExtraArgs -join ' ') > "$LogDir\serve.log" 2> "$LogDir\serve_err.log"
"$ft" serve --model-path "$Model" --port $Port $($ExtraArgs -join ' ') > "$LogDir\serve.log" 2> "$LogDir\serve_err.log"
"@
$runner = Join-Path $env:TEMP "freetoken_serve.cmd"
Set-Content $runner $cmd -Encoding ASCII

Write-Host "Starting '$(Split-Path $Model -Leaf)' on port $Port ..." -ForegroundColor Cyan
Start-Process cmd.exe -ArgumentList "/c", $runner -WindowStyle Hidden

for ($i = 1; $i -le 120; $i++) {
    Start-Sleep 5
    if (Select-String -Path "$LogDir\serve.log" -Pattern "ready to serve" -ErrorAction SilentlyContinue) {
        # web UI next to the API
        Start-Process cmd.exe -ArgumentList "/c", "cd /d `"$REPO\webui`" && python -m http.server 1420" -WindowStyle Hidden
        Write-Host ""
        Write-Host "  READY! Open http://localhost:1420 in your browser." -ForegroundColor Green
        Write-Host "  Logs: $LogDir\serve.log / serve_err.log"
        exit 0
    }
    # torch LOGS tracebacks as warnings and keeps going -- cpp_extension's
    # "Error checking compiler version" probe prints a full Traceback on every ROCm
    # start, and matching it aborted a load that went on to serve fine. Warning lines
    # carry torch's rank/severity stamp ("[rank0]:W0830 ..."); a real crash does not.
    $fatal = Select-String -Path "$LogDir\serve.log","$LogDir\serve_err.log" `
                 -Pattern "AssertionError|Traceback|exited during load" -ErrorAction SilentlyContinue |
             Where-Object { $_.Line -notmatch '\]:[WI]\d{4} ' }
    if ($fatal) {
        Write-Host "`n  The server hit an error while loading. Last lines:" -ForegroundColor Red
        Get-Content "$LogDir\serve_err.log","$LogDir\serve.log" -Tail 6 -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Host "." -NoNewline
}
Write-Host "`n  Timed out after 10 min - see $LogDir\serve_err.log" -ForegroundColor Red
