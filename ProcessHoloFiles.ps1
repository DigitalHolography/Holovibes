# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms

# Determine the script's directory
$scriptPath = Split-Path -Parent -Path $PSScriptRoot

# Function to select folder
function Select-Folder([string]$description, [string]$initial = '') {
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

# Prompt for folder containing .holo files (always first)
$holoFolderPath = Select-Folder -description "Select folder containing .holo files" 
if (-not $holoFolderPath) {
    Write-Host "No folder selected. Exiting script." -ForegroundColor Red
    exit
}

# Prompt for standard vs interactive configuration
Write-Host "Select configuration mode:" -ForegroundColor Cyan
Write-Host " 1) Standard config (all defaults)" -ForegroundColor Yellow
Write-Host " 2) Interactive setup" -ForegroundColor Yellow
$useStandardConfig = Read-Host "Enter choice (Press Enter for Standard)"

if ([string]::IsNullOrWhiteSpace($useStandardConfig) -or $useStandardConfig -eq '1') {
    # Standard defaults
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
} else {
    # Interactive setup
    function Select-File([string]$description, [string]$filter) {
        Write-Host $description -ForegroundColor Green
        $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
        $fileDialog.Filter = $filter
        $fileDialog.Title  = $description
        $fileDialog.InitialDirectory = $scriptPath
        if ($fileDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $fileDialog.FileName }
        return $null
    }

    function Select-OutputExtension {
        $options = @('.holo', '.mp4', '.avi')
        Write-Host "Select the output file extension:" -ForegroundColor Cyan
        for ($i = 0; $i -lt $options.Length; $i++) {
            Write-Host "  $($i+1). $($options[$i])" -ForegroundColor Yellow
        }
        $sel = Read-Host "Enter number (default 3 for .avi)"
        if ([string]::IsNullOrEmpty($sel) -or $sel -lt 1 -or $sel -gt $options.Length) { return '.avi' }
        return $options[$sel - 1]
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
    $modeChoice = Read-Host "Enter choice (default 1)"
    if ($modeChoice -notmatch '^[1-2]$') { $modeChoice = 1 }

    # Config files
    if ($modeChoice -eq 1) {
        $configFileNormal  = Get-ConfigFileOption
        $configFileMoments = $null
    } else {
        $configFileNormal  = $null
        $configFileMoments = Get-ConfigFileOption
    }

    # Frame skip and input fps
    $frameSkip   = Read-Host "Enter frame skip (default 8)";     if ($frameSkip -notmatch '^[0-9]+$') { $frameSkip = 8 }
    $input_fps   = Read-Host "Enter input fps (optional)";         if ($input_fps -notmatch '^[0-9]+$') { $input_fps = -1 }
    $outputExtension = Select-OutputExtension
}

# Find all .holo files
$holoFiles = Get-ChildItem -Path $holoFolderPath -Filter *.holo -Recurse
if (-not $holoFiles) {
    Write-Host "No .holo files found. Exiting." -ForegroundColor Red
    exit
}

# Determine executable path
$exePath = if (Test-Path 'Holovibes.exe') { 'Holovibes.exe' } else { 'build/bin/Holovibes.exe' }
Write-Host "Using executable: $exePath" -ForegroundColor Cyan

# Function to run Holovibes
function Execute-Holovibes {
    param($inputFile, $outputFile, $skip, $fps, $config, $moments)
    $args = "-i `"$inputFile`" -o `"$outputFile`""
    if ($fps -ne -1) { $args += " -f $fps" }
    if ($moments)    { $args += ' --moments_record' } else { $args += " --frame_skip $skip" }
    if ($config)     { $args += " -c `"$config`"" }
    Write-Host "Running: $exePath $args" -ForegroundColor Yellow
    Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -Wait
}

# Process each file
foreach ($file in $holoFiles) {
    $in   = $file.FullName
    $base = $file.BaseName
    if ($modeChoice -eq 1) {
        $out = Join-Path $holoFolderPath "$base$outputExtension"
        Execute-Holovibes $in $out $frameSkip $input_fps $configFileNormal $false
    } else {
        $out = Join-Path $holoFolderPath "${base}_moments.holo"
        Execute-Holovibes $in $out $frameSkip $input_fps $configFileMoments $true
    }
}

Write-Host "Processing complete." -ForegroundColor Green
