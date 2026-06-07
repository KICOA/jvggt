# Run demo_jvggt with Visual Studio MSVC (fixes Jittor mspdbcore.dll / cl compile errors).
# Usage: .\run_demo_jvggt.ps1
#        .\run_demo_jvggt.ps1 --inference_only

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArgs
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Find-VsDevShell {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $install = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $install) { return $null }
    $ps1 = Join-Path $install "Common7\Tools\Launch-VsDevShell.ps1"
    if (Test-Path $ps1) { return $ps1 }
    return $null
}

# Remove Jittor bundled MSVC (optional; avoids broken mspdbcore in cache copy)
$jittorMsvc = Join-Path $env:USERPROFILE ".cache\jittor\msvc"
if (Test-Path $jittorMsvc) {
    Write-Host "Removing bundled Jittor MSVC cache: $jittorMsvc"
    Remove-Item -Recurse -Force $jittorMsvc
}

$devShell = Find-VsDevShell
if ($devShell) {
    Write-Host "Loading Visual Studio dev environment (x64) ..."
    Import-Module $devShell -ArgumentList @("-Arch", "amd64", "-SkipAutomaticLocation") -DisableNameChecking
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) {
        $env:cc_path = $cl.Source
        Write-Host "cc_path=$($env:cc_path)"
    }
} else {
    Write-Host "WARNING: vswhere / VsDevShell not found. Install VS 2022 Build Tools (C++ workload)."
}

Set-Location $Root
python (Join-Path $Root "demo_jvggt.py") @DemoArgs
