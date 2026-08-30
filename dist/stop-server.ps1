# Stop this repo's FreeToken engine, its worker processes, and the web UI.
$REPO = Split-Path -Parent $PSScriptRoot
$repoPattern = [regex]::Escape($REPO)
$processes = @(Get-CimInstance Win32_Process)
$serverIds = @()
$serverErrorLog = Join-Path $env:TEMP "freetoken-logs\serve_err.log"
if (Test-Path $serverErrorLog) {
    $serverErrorContent = Get-Content $serverErrorLog -Raw
    if ($serverErrorContent) {
        $serverIds = @([regex]::Matches(
            $serverErrorContent,
            "Started server process \[(\d+)\]"
        ) | ForEach-Object { [int]$_.Groups[1].Value })
    }
}
$orphanPattern = if ($serverIds.Count -gt 0) {
    "parent_pid=(" + ($serverIds -join "|") + ")\b"
} else { $null }
$targets = @($processes | Where-Object {
    $_.ExecutablePath -like "$REPO\*" -or
    $_.CommandLine -match $repoPattern -or
    $_.CommandLine -match "http\.server\s+1420" -or
    $serverIds -contains $_.ProcessId -or
    $serverIds -contains $_.ParentProcessId -or
    ($orphanPattern -and $_.CommandLine -match $orphanPattern)
})

# Multiprocessing workers may only reference their parent PID, not the repo path.
do {
    $targetIds = @($targets.ProcessId)
    $children = @($processes | Where-Object {
        $targetIds -contains $_.ParentProcessId -and $targetIds -notcontains $_.ProcessId
    })
    $targets += $children
} while ($children.Count -gt 0)

$targets | Sort-Object ProcessId -Unique | ForEach-Object {
    try {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
        "stopped $($_.ProcessId) $($_.Name)"
    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
        # A parent process can terminate its workers before their turn here.
    }
}
