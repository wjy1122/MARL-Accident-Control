$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Jiayi\Projects\MARL-Accident-Control-V2'
$py = Join-Path $root '.venv\Scripts\python.exe'
$report = Join-Path $root 'phase0_env_report.txt'
Set-Location $root

function Append-Section([string]$title) {
    Add-Content -Path $report -Value ""
    Add-Content -Path $report -Value "=== $title ==="
}

# Fresh report
@"
=== phase0 environment report ===
Generated: $(Get-Date -Format o)
Project: $root
"@ | Set-Content -Path $report -Encoding UTF8

Append-Section 'pip install -r requirements.txt'
& $py -m pip install -r requirements.txt 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'venv python --version'
& $py --version 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'torch version'
& $py -c "import torch; print(torch.__version__)" 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'torch cuda'
& $py -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'numpy version'
& $py -c "import numpy; print(numpy.__version__)" 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'system python --version'
python --version 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'sumo --version'
sumo --version 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'netconvert --version'
netconvert --version 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'TraCI'
$traciCode = @'
import sys, os
sumo_home = os.environ.get("SUMO_HOME", "")
print("SUMO_HOME=", sumo_home)
tools = os.path.join(sumo_home, "tools") if sumo_home else ""
if tools:
    sys.path.insert(0, tools)
try:
    import traci
    print("TraCI OK")
except Exception as e:
    print("TraCI failed:", repr(e))
    candidates = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Program Files (x86)\SUMO",
        r"C:\Program Files\SUMO",
    ]
    # Also try dirname of sumo.exe on PATH
    import shutil
    sumo_exe = shutil.which("sumo")
    if sumo_exe:
        # e.g. ...\Sumo\bin\sumo.exe -> ...\Sumo
        candidates.insert(0, os.path.dirname(os.path.dirname(sumo_exe)))
    for c in candidates:
        tools2 = os.path.join(c, "tools")
        if not os.path.isdir(tools2):
            continue
        print("Trying SUMO_HOME=", c)
        os.environ["SUMO_HOME"] = c
        if tools2 not in sys.path:
            sys.path.insert(0, tools2)
        try:
            if "traci" in sys.modules:
                del sys.modules["traci"]
            import traci
            print("TraCI OK with", c)
            break
        except Exception as e2:
            print("Failed", c, repr(e2))
    else:
        print("TraCI still unavailable after common path retries")
'@
& $py -c $traciCode 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Append-Section 'old project venv'
if (Test-Path 'C:\Users\Jiayi\Projects\MARL-Accident-Control\.venv') {
    Add-Content -Path $report -Value 'OLD_VENV_EXISTS'
} else {
    Add-Content -Path $report -Value 'OLD_VENV_MISSING'
}

Append-Section 'tests\test_project.py'
& $py tests\test_project.py 2>&1 | Out-String | Add-Content -Path $report -Encoding UTF8

Add-Content -Path $report -Value ""
Add-Content -Path $report -Value "=== DONE ==="
Write-Output "REPORT_WRITTEN:$report"
