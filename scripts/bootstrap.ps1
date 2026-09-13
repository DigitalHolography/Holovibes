# Dot-source this script to initialize the current terminal:
#   . .\scripts\bootstrap.ps1
[CmdletBinding()]
param([switch]$DependenciesOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$config = Get-Content -LiteralPath (Join-Path $repoRoot 'cmake/toolchain.json') -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath (Join-Path $repoRoot 'vcpkg.json') -Raw | ConvertFrom-Json

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio 2022 with the components in .vsconfig first.'
}
$instances = & $vswhere -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
$vsPath = $null
foreach ($instance in $instances) {
    $toolsets = Get-ChildItem -LiteralPath (Join-Path $instance.installationPath 'VC/Tools/MSVC') -Directory
    if ($toolsets.Name -like "$($config.msvcToolset).*") { $vsPath = $instance.installationPath; break }
}
if (-not $vsPath) { throw "Install MSVC $($config.msvcToolset) using Visual Studio Installer (.vsconfig)." }

$sdkRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits/10/Include/$($config.windowsSdk)"
if (-not (Test-Path -LiteralPath $sdkRoot)) { throw "Install Windows SDK $($config.windowsSdk) (.vsconfig)." }
Import-Module (Join-Path $vsPath 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=$($config.msvcToolset) -winsdk=$($config.windowsSdk)"
# Keep CMake's localized /showIncludes detection and Ninja's compiler output in
# the same encoding, including on machines without the English VS language pack.
Invoke-Checked chcp @('65001')
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:VCPKG_VISUAL_STUDIO_PATH = $vsPath

$cmakeTools = Join-Path $vsPath 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$env:PATH = "$(Join-Path $cmakeTools 'CMake/bin');$(Join-Path $cmakeTools 'Ninja');$env:PATH"
$cmakeVersion = (& cmake --version | Select-Object -First 1) -replace '^cmake version ([0-9.]+).*', '$1'
if ([version]$cmakeVersion -lt [version]$config.cmakeMinimum) { throw "CMake $($config.cmakeMinimum)+ is required. Update the Visual Studio CMake component." }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'Install the Visual Studio CMake/Ninja component (.vsconfig).' }

if (-not $DependenciesOnly) {
    $cudaCandidates = @($env:CUDA_PATH)
    $cudaBase = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA'
    if (Test-Path -LiteralPath $cudaBase) {
        $cudaCandidates += Get-ChildItem -LiteralPath $cudaBase -Directory |
            Where-Object Name -Match '^v\d+\.\d+$' |
            Sort-Object { [version]($_.Name.Substring(1)) } -Descending |
            Select-Object -ExpandProperty FullName
    }
    $selectedCuda = $null
    foreach ($candidate in $cudaCandidates) {
        if (-not $candidate) { continue }
        $nvcc = Join-Path $candidate 'bin/nvcc.exe'
        if (-not (Test-Path -LiteralPath $nvcc)) { continue }
        $nvccVersion = (& $nvcc --version) -join ' '
        if ($LASTEXITCODE -eq 0 -and $nvccVersion -match 'release (\d+\.\d+)' -and [version]$Matches[1] -ge [version]$config.cudaMinimum) {
            $selectedCuda = $candidate; break
        }
    }
    if (-not $selectedCuda) { throw "Install CUDA Toolkit $($config.cudaMinimum)+, or set CUDA_PATH to it. Use -DependenciesOnly to validate third-party libraries without CUDA." }
    $env:CUDA_PATH = $selectedCuda
    $env:CUDACXX = Join-Path $selectedCuda 'bin/nvcc.exe'
    $env:PATH = "$(Join-Path $selectedCuda 'bin');$(Join-Path $selectedCuda 'bin/x64');$env:PATH"
}

$vcpkgRoot = Join-Path $repoRoot 'out/tools/vcpkg'
$baseline = $manifest.'builtin-baseline'
if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot '.git'))) {
    if (Test-Path -LiteralPath $vcpkgRoot) { throw "$vcpkgRoot exists but is not a Git checkout." }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $vcpkgRoot) | Out-Null
    Invoke-Checked git @('clone', '--filter=blob:none', 'https://github.com/microsoft/vcpkg.git', $vcpkgRoot)
}
$vcpkgHead = & git -C $vcpkgRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the vcpkg checkout.' }
if ($vcpkgHead -ne $baseline) {
    $changes = & git -C $vcpkgRoot status --porcelain
    if ($LASTEXITCODE -ne 0 -or $changes) { throw 'The local vcpkg checkout has changes. Restore them before changing its pinned revision.' }
    Invoke-Checked git @('-C', $vcpkgRoot, 'fetch', 'origin', $baseline)
    Invoke-Checked git @('-C', $vcpkgRoot, 'checkout', '--detach', $baseline)
}
if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot 'vcpkg.exe')) -or $vcpkgHead -ne $baseline) {
    Invoke-Checked (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') @('-disableMetrics')
}
$env:VCPKG_ROOT = $vcpkgRoot
$env:VCPKG_OVERLAY_TRIPLETS = Join-Path $repoRoot 'cmake/triplets'
# The normal local cache remains enabled. This optional source is read-only on developer machines.
if ($env:HOLOVIBES_BINARY_CACHE) {
    $cacheSource = $env:HOLOVIBES_BINARY_CACHE
    $existingSources = @($env:VCPKG_BINARY_SOURCES -split ';')
    if ($cacheSource -notin $existingSources) {
        $env:VCPKG_BINARY_SOURCES = (@($existingSources | Where-Object { $_ }) + $cacheSource) -join ';'
    }
}
Write-Host "Ready: MSVC $env:VCToolsVersion, SDK $($config.windowsSdk), CMake $cmakeVersion, vcpkg $baseline"
