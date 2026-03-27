# Windows Setup for vLLM CPU Server

This project includes Windows-friendly launchers that call the existing `scripts/start_vllm_server_cpu.sh` through Git Bash.

## Prerequisites

- Windows 10 or later
- Git for Windows installed
- PowerShell 5.1+ or PowerShell 7+

## Three ways to start

### 1) Double-click the batch file

Run:

```text
scripts/start_vllm_server_cpu.bat
```

What it does:
- Changes to the repository root
- Automatically finds Git Bash in the standard install locations
- Runs the CPU server shell script

### 2) Run the PowerShell launcher

From PowerShell:

```powershell
.\scripts\start_vllm_server_cpu.ps1
```

Or run it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_vllm_server_cpu.ps1
```

If execution policy blocks the script, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_vllm_server_cpu.ps1
```

### 3) Call the shell script manually from Git Bash

Open Git Bash in the repository root and run:

```bash
./scripts/start_vllm_server_cpu.sh
```

## Screenshot examples

If you add screenshots later, place them here:

- Launcher in File Explorer
- PowerShell startup output
- Server running successfully

If you can include real screenshots, add them to `docs/images/` and link them here.

Suggested file names:

- `docs/images/windows-bat-launch.png`
- `docs/images/windows-powershell-launch.png`
- `docs/images/windows-server-running.png`

## Troubleshooting

### Git Bash not found

Error:

```text
Git Bash not found. Please install Git for Windows.
```

Fix:
- Install Git for Windows
- Verify Bash exists in one of these paths:
  - `C:\Program Files\Git\bin\bash.exe`
  - `C:\Program Files (x86)\Git\bin\bash.exe`

### PowerShell script blocked

Fix:
- Run `Set-ExecutionPolicy -Scope Process Bypass`
- Then rerun the `.ps1` script

### Server does not start

Check:
- You are running from the repository root
- `scripts/start_vllm_server_cpu.sh` exists
- Python dependencies for CPU inference are installed
- The model/config in `.env.vllm.cpu` is correct

### Port already in use

Fix:
- Stop the process using the same port
- Or change the server port in the CPU launch configuration

### Permission or path issues

Fix:
- Prefer the `.bat` or `.ps1` launcher from the repo root
- Avoid running from a moved/copied `scripts` folder
