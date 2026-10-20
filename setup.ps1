# setup.ps1 - one-click environment setup for the RAG PDF reader (Windows)
# Launched by setup.bat. Everything is also written to setup_log.txt.
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot 'setup_log.txt'
Set-Content -LiteralPath $log -Value "=== Setup started $(Get-Date) ==="

function Log([string]$m) { Write-Host $m; Add-Content -LiteralPath $log -Value $m }

function Run([string]$exe, [string[]]$a) {
    Log ("> " + $exe + " " + ($a -join ' '))
    & $exe @a 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; Add-Content -LiteralPath $log -Value $s }
    return $LASTEXITCODE
}

# ---------- 1. Diagnostics ----------
Log "`n=== [1/5] Python installations found ==="
if (Get-Command py -ErrorAction SilentlyContinue) { Run 'py' @('-0p') | Out-Null } else { Log "(py launcher not found)" }
if (Get-Command python -ErrorAction SilentlyContinue) { Run 'python' @('--version') | Out-Null } else { Log "(python not on PATH)" }
if (Get-Command pip -ErrorAction SilentlyContinue) { Run 'pip' @('--version') | Out-Null } else { Log "(pip not on PATH)" }

# ---------- 2. Pick a Python >= 3.10 ----------
Log "`n=== [2/5] Choosing a Python >= 3.10 ==="
$candidates = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in '3.12', '3.13', '3.11', '3.10', '3.14') { $candidates += , @('py', "-$v") }
}
if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @('python') }

$pyExe = $null
foreach ($c in $candidates) {
    $exe = $c[0]; $pre = @(); if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
    $out = & $exe @pre -c "import sys; print(sys.executable) if sys.version_info >= (3,10) else sys.exit(1)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { $pyExe = "$out".Trim(); break }
}
if (-not $pyExe) {
    Log "ERROR: No Python 3.10+ found."
    Log "Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run setup.bat again."
    exit 1
}
Log "Using: $pyExe"
Run $pyExe @('--version') | Out-Null

# ---------- 3. Virtual environment ----------
Log "`n=== [3/5] Creating virtual environment .venv ==="
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    if ((Run $pyExe @('-m', 'venv', '.venv')) -ne 0) { Log "ERROR: could not create .venv"; exit 1 }
} else { Log ".venv already exists, reusing it" }

# ---------- 4. Install packages ----------
Log "`n=== [4/5] Installing packages (this can take several minutes: PyTorch is large) ==="
Run $venvPy @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null
if ((Run $venvPy @('-m', 'pip', 'install', '-r', 'requirements.txt')) -ne 0) { Log "ERROR: pip install failed"; exit 1 }

# ---------- 5. Verify ----------
Log "`n=== [5/5] Verifying packages, embedding model and API keys ==="
# Pick up keys saved with setx even if this window started before they were set
foreach ($k in 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY') {
    if (-not [Environment]::GetEnvironmentVariable($k, 'Process')) {
        $u = [Environment]::GetEnvironmentVariable($k, 'User')
        if ($u) { [Environment]::SetEnvironmentVariable($k, $u, 'Process') }
    }
}
$rc = Run $venvPy @('check_setup.py')
if ($rc -eq 0) { Log "`n=== SETUP COMPLETE ===`nRun the program with:  .\run.bat --provider claude" }
else { Log "`n=== SETUP FINISHED WITH PROBLEMS (see messages above) ===" }
