# ============================================================
#  Stop any running FreeToken engine / web UI -- and their children.
#
#  Matching command lines alone is not enough. The engine's scheduler runs in
#  multiprocessing SPAWN workers, whose command line is
#      python.exe -c "from multiprocessing.spawn import spawn_main; ..."
#  which names neither `serve` nor `freetoken` nor the model. Killing only the
#  roots left those workers alive holding ~20 GiB of pinned expert banks AND the
#  inherited serve.log / serve_err.log handles, which then trips run-server.ps1's
#  "log is still locked" guard on the next launch.
#
#  So: kill the roots' whole process tree, and separately sweep up spawn workers
#  already orphaned by an earlier partial stop. Those are identified by their
#  EXECUTABLE, not their command line -- only this repo's .venv python is ours, so
#  an unrelated multiprocessing app on the box is never touched.
# ============================================================
$REPO = Split-Path -Parent $PSScriptRoot
$snapshot = Get-CimInstance Win32_Process |
            Select-Object ProcessId, ParentProcessId, Name, CommandLine, CreationDate, ExecutablePath

$isOurs = { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($REPO, 'OrdinalIgnoreCase') }

$roots = $snapshot | Where-Object {
    ($_.Name -eq 'ft.exe' -or $_.Name -like '*python*') -and
    $_.CommandLine -match 'serve|http\.server 1420|freetoken'
}
# Workers orphaned by a previous run: our venv's python, sitting in a spawn loop.
$orphans = $snapshot | Where-Object {
    $_.CommandLine -match 'multiprocessing\.spawn' -and (& $isOurs)
}

# Children of a pid, transitively. A dead PID gets reused, so a stale ParentProcessId
# can point at an unrelated live process -- only follow a child that started after its
# claimed parent did.
function Get-Descendants($processId, $startedAt) {
    $kids = $snapshot | Where-Object {
        $_.ParentProcessId -eq $processId -and $_.ProcessId -ne $processId -and
        (-not $startedAt -or -not $_.CreationDate -or $_.CreationDate -ge $startedAt)
    }
    foreach ($k in $kids) { $k; Get-Descendants $k.ProcessId $k.CreationDate }
}

$targets = @()
foreach ($r in @($roots) + @($orphans)) {
    $targets += $r
    $targets += Get-Descendants $r.ProcessId $r.CreationDate
}
$targets = @($targets | Sort-Object ProcessId -Unique)
if (-not $targets) { "no FreeToken processes running"; return }

# Depth in the kill set, so we can go deepest-first: a parent killed early can spawn a
# replacement worker on its way out.
$depth = @{}
foreach ($t in $targets) {
    $d = 0; $cur = $t
    while ($cur -and $d -lt 32) {
        $cur = @($targets | Where-Object { $_.ProcessId -eq $cur.ParentProcessId })[0]
        if ($cur) { $d++ }
    }
    $depth[[int]$t.ProcessId] = $d
}
foreach ($t in ($targets | Sort-Object { -$depth[[int]$_.ProcessId] })) {
    $mb = 0
    try { $mb = [int]((Get-Process -Id $t.ProcessId -ErrorAction Stop).WorkingSet64 / 1MB) } catch {}
    try {
        Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop
        "stopped $($t.ProcessId) $($t.Name) ($mb MB)"
    } catch {
        # Usually already gone because we killed its parent -- that is the point.
        if (Get-Process -Id $t.ProcessId -ErrorAction SilentlyContinue) {
            Write-Warning "could not stop $($t.ProcessId) $($t.Name): $($_.Exception.Message)"
        }
    }
}
