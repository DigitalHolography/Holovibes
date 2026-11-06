param(
    [switch]$Lsci,
    [string]$InputFolder,
    [string]$OutputFolder
)

$ErrorActionPreference = 'Stop'

# Determine directories relative to this script
$scriptDir  = $PSScriptRoot
$scriptPath = Split-Path -Parent -Path $scriptDir  # install root if script lives in "scripts/"

function Get-PresetPath([string]$name) {
    $candidates = @(
        (Join-Path $scriptDir "Ui/assets/preset/$name")
        (Join-Path $scriptDir "AppData/preset/$name")
        (Join-Path $scriptPath "AppData/preset/$name")
    )
    $path = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $path) { throw "Preset not found: $name" }
    return $path
}

# Function to select folder
function Select-Folder([string]$description, [string]$initial = '') {
    Add-Type -AssemblyName System.Windows.Forms
    Write-Host $description -ForegroundColor Green
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.Description = $description
    if ($initial -and (Test-Path $initial)) {
        $folderBrowser.SelectedPath = $initial
    } else {
        $folderBrowser.SelectedPath = [Environment]::GetFolderPath('MyDocuments')
    }
    if ($folderBrowser.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $folderBrowser.SelectedPath
    }
    return $null
}

# Prompt for folder containing .holo files unless supplied on the command line.
$holoFolderPath = $InputFolder
if (-not $holoFolderPath) {
    $holoFolderPath = Select-Folder -description "Select folder containing .holo files"
}
if (-not $holoFolderPath) {
    Write-Host "No folder selected. Exiting script." -ForegroundColor Red
    exit
}
if (-not (Test-Path -LiteralPath $holoFolderPath -PathType Container)) {
    throw "Input folder not found: $holoFolderPath"
}
$holoFolderPath = (Resolve-Path -LiteralPath $holoFolderPath).Path
if ($OutputFolder) {
    $OutputFolder = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputFolder)
    New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
}

# Prompt for standard, interactive, or LSCI configuration.
if ($Lsci) {
    $useStandardConfig = '3'
} else {
    Write-Host "Select configuration mode:" -ForegroundColor Cyan
    Write-Host " 1) Standard config (all defaults)" -ForegroundColor Yellow
    Write-Host " 2) Interactive setup" -ForegroundColor Yellow
    Write-Host " 3) LSCI AVI (registration on, all contrast maps)" -ForegroundColor Yellow
    $useStandardConfig = Read-Host "Enter choice (Press Enter for Standard)"
}

if ($useStandardConfig -eq '3') {
    $modeChoice      = 5
    $frameSkip       = 0
    $input_fps       = -1
    $outputExtension = '.avi'
    $configFileNormal = Get-PresetPath 'settingsLSCI.json'
    $configFileMoments = $null
    Write-Host "LSCI AVI: 24 fps playback, 5 x 5 pixels, 16-frame windows, registration on, no frame skipping, contrast fixed from the first map." -ForegroundColor Cyan
}
elseif ([string]::IsNullOrWhiteSpace($useStandardConfig) -or $useStandardConfig -eq '1') {
    # --- Standard defaults
    Write-Host "Using standard configuration. Processing with defaults..." -ForegroundColor Cyan
    $modeChoice      = 1
    $frameSkip       = 8
    $input_fps       = -1
    $outputExtension = '.avi'

    # Default preset configuration file
    $presetPath = "AppData\preset\settingsDoppler37kHz.json"
    if (Test-Path $presetPath) {
        $configFileNormal = $presetPath
        Write-Host "Using preset config: $presetPath" -ForegroundColor Green
    } else {
        $configFileNormal = $null
        Write-Host "Preset config not found. Skipping config file." -ForegroundColor Yellow
    }
    $configFileMoments = $null
}
else {
    # --- Interactive setup ---
    function Select-File([string]$description, [string]$filter) {
        Add-Type -AssemblyName System.Windows.Forms
        Write-Host $description -ForegroundColor Green
        $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
        $fileDialog.Filter = $filter
        $fileDialog.Title  = $description
        $fileDialog.InitialDirectory = $scriptPath
        if ($fileDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $fileDialog.FileName }
        return $null
    }

    function Select-OutputExtension {
    param([int]$modeChoice)

    if ($modeChoice -eq 5) { return '.avi' }

    if ($modeChoice -eq 2 -or $modeChoice -eq 3 -or $modeChoice -eq 4) {
        Write-Host "For Moments and OCT/OCT_FLOAT records, output extension is fixed to .h5" -ForegroundColor Cyan
        return '.h5'
    }
    else {
        $options = @('.holo', '.mp4', '.avi')
        Write-Host "Select the output file extension:" -ForegroundColor Cyan
        for ($i = 0; $i -lt $options.Length; $i++) {
            Write-Host "  $($i+1). $($options[$i])" -ForegroundColor Yellow
        }
        $sel = Read-Host "Enter number (default 3 for .avi)"
        if ([string]::IsNullOrEmpty($sel) -or $sel -lt 1 -or $sel -gt $options.Length) { return '.avi' }
        return $options[$sel - 1]
    }
}

    function Get-ConfigFileOption {
        Write-Host "Select the configuration file option:" -ForegroundColor Cyan
        Write-Host " 1. Use standard preset" -ForegroundColor Yellow
        Write-Host " 2. Skip config file" -ForegroundColor Yellow
        Write-Host " 3. Browse for config file" -ForegroundColor Yellow
        $choice = Read-Host "Enter choice (default 1)"
        if ([string]::IsNullOrEmpty($choice) -or $choice -notmatch '^[1-3]$') { $choice = '1' }
        switch ($choice) {
            '1' {
                $path = "AppData\preset\settingsDoppler37kHz.json"
                if (Test-Path $path) { Write-Host "Using preset: $path" -ForegroundColor Green; return $path }
                Write-Host "Preset not found. Skipping." -ForegroundColor Yellow
                return $null
            }
            '2' {
                Write-Host "No config file selected." -ForegroundColor Yellow
                return $null
            }
            '3' {
                $fp = Select-File -description 'Select a configuration file' -filter 'JSON Files (*.json)|*.json|All Files (*.*)|*.*'
                if ($fp) { Write-Host "Selected: $fp" -ForegroundColor Green; return $fp }
                Write-Host "No file selected. Skipping." -ForegroundColor Yellow
                return $null
            }
        }
    }

    # Recording mode 
    Write-Host "Select recording mode:" -ForegroundColor Cyan
    Write-Host " 1. Image rendering" -ForegroundColor Yellow
    Write-Host " 2. Statistical moments" -ForegroundColor Yellow
    Write-Host " 3. OCT" -ForegroundColor Yellow
    Write-Host " 4. OCT_FLOAT" -ForegroundColor Yellow
    Write-Host " 5. LSCI AVI (registration on, all contrast maps)" -ForegroundColor Yellow
    $modeChoice = Read-Host "Enter choice (default 1)"
    if ($modeChoice -notmatch '^[1-5]$') { $modeChoice = 1 }

    # Config files
    if ($modeChoice -eq 5) {
        $configFileNormal = Get-PresetPath 'settingsLSCI.json'
        $configFileMoments = $null
    }
    elseif ($modeChoice -eq 1) {
        $configFileNormal  = Get-ConfigFileOption
        $configFileMoments = $null
    }
    elseif ($modeChoice -eq 2) {
        $configFileNormal  = $null
        $configFileMoments = Get-ConfigFileOption
    }
    else {
        $configFileNormal  = Get-ConfigFileOption
        $configFileMoments = $null
    }

    # Frame skip et input fps
    if ($modeChoice -eq 5) {
        $frameSkip = 0
        Write-Host "LSCI uses registration and saves every complete 16-frame contrast map as AVI at 24 fps playback." -ForegroundColor Cyan
    } else {
        $frameSkip = Read-Host "Enter frame skip (default 8)"
        if ($frameSkip -notmatch '^[0-9]+$') { $frameSkip = 8 }
    }
    $input_fps       = Read-Host "Enter input fps (optional)";    if ($input_fps -notmatch '^[0-9]+$') { $input_fps = -1 }
    $outputExtension = Select-OutputExtension -modeChoice $modeChoice
}

# Find all .holo files
$holoFiles = Get-ChildItem -LiteralPath $holoFolderPath -Filter *.holo -Recurse -File
if (-not $holoFiles) {
    Write-Host "No .holo files found. Exiting." -ForegroundColor Red
    exit
}

# Determine executable path relative to the script location first, then fall back to CWD
$exeCandidates = @(
    $env:HOLOVIBES_BIN
    (Join-Path $scriptDir 'Holovibes.exe')
    (Join-Path $scriptDir 'out/build/windows-dev/Holovibes.exe')
    (Join-Path $scriptDir 'out/build/windows-release/Holovibes.exe')
    (Join-Path $scriptPath 'Holovibes.exe')
    (Join-Path $scriptPath 'build/bin/Holovibes.exe')
    'Holovibes.exe'
    'build/bin/Holovibes.exe'
)

$exePath = $exeCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $exePath) {
    throw "Unable to locate Holovibes.exe. Build the application or set HOLOVIBES_BIN."
}
$exePath = (Resolve-Path -LiteralPath $exePath).Path
Write-Host "Using executable: $exePath" -ForegroundColor Cyan

# Function to run Holovibes 
function Execute-Holovibes {
    param(
        $inputFile,
        $outputFile,
        $skip,
        $fps,
        $config,
        $mode  # 1=image, 2=moments, 3=OCT, 4=OCT_FLOAT, 5=LSCI AVI
    )
    $commandArguments = "-i `"$inputFile`" -o `"$outputFile`""
    if ($fps -ne -1) { $commandArguments += " -f $fps" }

    switch ($mode) {
        1 { $commandArguments += " --frame_skip $skip" }
        2 { $commandArguments += " --moments_record" }
        3 { $commandArguments += " --oct_cube_record" }
        4 { $commandArguments += " --oct_cube_float_record" }
        5 { $commandArguments += " --frame_skip 0 --noskip_acc --mp4_fps 24" }
    }

    if ($config) { $commandArguments += " -c `"$config`"" }

    if ([IO.Path]::GetExtension($outputFile) -ieq '.avi' -and ($mode -eq 1 -or $mode -eq 5)) {
        $commandArguments += " --contrast_once"
        Write-Host "Display contrast will be estimated from the first complete output frame and held fixed for this AVI." -ForegroundColor Cyan
    }

    Write-Host "Running: $exePath $commandArguments" -ForegroundColor Yellow
    $process = Start-Process -FilePath $exePath -ArgumentList $commandArguments -NoNewWindow -Wait -PassThru `
        -WorkingDirectory (Split-Path -Parent $exePath)
    if ($process.ExitCode -ne 0) {
        throw "Holovibes failed for '$inputFile' (exit $($process.ExitCode))."
    }
}

# Process each file
foreach ($file in $holoFiles) {
    $in   = $file.FullName
    $base = $file.BaseName
    $outDir = $file.DirectoryName  # dossier d'origine du fichier
    if ($OutputFolder) {
        $relativeDir = $file.DirectoryName.Substring($holoFolderPath.TrimEnd('\', '/').Length).TrimStart('\', '/')
        $outDir = Join-Path $OutputFolder $relativeDir
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    switch ($modeChoice) {
        1 {
            $out = Join-Path $outDir "${base}_p${outputExtension}"
        }
        2 {
            $out = Join-Path $outDir "${base}_moments.h5"
        }
        3 {
            $out = Join-Path $outDir "${base}_oct.h5"
        }
        4 {
            $out = Join-Path $outDir "${base}_oct_float.h5"
        }
        5 {
            $out = Join-Path $outDir "${base}_lsci.avi"
        }
    }

    Execute-Holovibes $in $out $frameSkip $input_fps $configFileNormal $modeChoice
}

Write-Host "Processing complete." -ForegroundColor Green
