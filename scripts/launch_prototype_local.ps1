<#
.SYNOPSIS
    ARX Terminal — UX Research Programme: Authenticated Local Research Server Launcher
.DESCRIPTION
    Launches the authenticated local research server (research_server.py) on 127.0.0.1
    (loopback only), establishes a moderator bootstrap authorization context,
    and opens the default browser to a clean URL (http://127.0.0.1:3456/prototype/).
.PARAMETER Port
    Local TCP port to bind (default: 3456)
#>
param(
    [int]$Port = 3456
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$harnessOut = Join-Path $repoRoot "prototype-harness\out"
$serverScript = Join-Path $repoRoot "scripts\research_server.py"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " ARX TERMINAL — UX RESEARCH PROGRAMME AUTHENTICATED LOCAL LAUNCHER" -ForegroundColor Cyan
Write-Host " Defect Remediation: ARX-UX-P00-INSTR-001 (Durable Local Capture & Custody)" -ForegroundColor Gray
Write-Host " Research Owner: David Akara | Governance: ARX Closure Model v2" -ForegroundColor Gray
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Verify files exist
if (-not (Test-Path $harnessOut)) {
    Write-Warning "[WARN] Pre-built prototype export not found at: $harnessOut. Static assets may fallback."
}
if (-not (Test-Path $serverScript)) {
    Write-Error "[FAIL] Research server script not found at: $serverScript"
}

# 2. Check port availability
$tcpConn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($tcpConn) {
    Write-Warning "[WARN] Port $Port is currently in use. Attempting to use port 3458..."
    $Port = 3458
}

# 3. Generate high-entropy 32-byte hex moderator bootstrap token
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 32
$rng.GetBytes($bytes)
$bootstrapToken = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""

$url = "http://127.0.0.1:$Port/prototype/"

Write-Host "[1/3] Starting authenticated loopback research server on 127.0.0.1:$Port..." -ForegroundColor Green
Write-Host "      Security: Loopback-only binding. Moderator bootstrap authentication active." -ForegroundColor DarkGray
Write-Host "      Bearer secrets eliminated from URLs. Host and origin protection enabled." -ForegroundColor DarkGray

# 4. Spawn Python research server with bootstrap token
$serverProcess = Start-Process -FilePath "python" `
    -ArgumentList "$serverScript", "--port", "$Port", "--bootstrap-token", "$bootstrapToken" `
    -PassThru `
    -NoNewWindow

Start-Sleep -Milliseconds 1200

if ($serverProcess.HasExited) {
    Write-Error "[FAIL] Research server process terminated unexpectedly."
}

Write-Host "[2/3] Authenticated Research Server is active!" -ForegroundColor Green
Write-Host "      Clean Prototype URL: $url" -ForegroundColor Yellow
Write-Host "      MODERATOR BOOTSTRAP PASSKEY: $bootstrapToken" -ForegroundColor Magenta
Write-Host "      (Use this passkey in the browser prompter to enroll the session)" -ForegroundColor DarkGray

# 5. Open browser with clean URL
Write-Host "[3/3] Opening default browser to clean local prototype URL..." -ForegroundColor Green
Start-Process $url

Write-Host ""
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host " PROTOTYPE RESEARCH PROTOCOL INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host " 1. Browser will load cleanly at $url (no credentials in URL/logs)." -ForegroundColor White
Write-Host " 2. If prompted for Moderator Authorization, enter the bootstrap passkey above." -ForegroundColor White
Write-Host " 3. Select 'P00-SYNTHETIC (Automated Rehearsal)' for nonhuman synthetic execution." -ForegroundColor White
Write-Host " 4. When all trials complete, the harness automatically transmits telemetry" -ForegroundColor White
Write-Host "    to /api/research/custody/session_ingest and verifies server-signed receipt." -ForegroundColor White
Write-Host " 5. UI verifies HMAC signature and renders verified sealed completion badge." -ForegroundColor White
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host " Press Ctrl+C in this PowerShell terminal to stop the local research server." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

try {
    # Keep process running until interrupted
    $serverProcess.WaitForExit()
}
finally {
    if (-not $serverProcess.HasExited) {
        Write-Host "`nStopping research server..." -ForegroundColor Gray
        Stop-Process -Id $serverProcess.Id -Force
        Write-Host "[OK] Research server terminated cleanly." -ForegroundColor Green
    }
}
