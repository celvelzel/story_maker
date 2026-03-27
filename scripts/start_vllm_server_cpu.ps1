# Start vLLM CPU server via Git Bash
$gitBash = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $gitBash) {
    Write-Error "Git Bash not found. Please install Git for Windows."
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "start_vllm_server_cpu.sh"
& $gitBash --login $scriptPath
