# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms

# Function to prompt the user to select a file, starting at the script's location
function Select-File([string]$description, [string]$filter) {
    Write-Host $description -ForegroundColor Green
    $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $fileDialog.Filter = $filter
    $fileDialog.Title = $description
    
    # Set InitialDirectory to the folder where the script is executed
    $scriptPath = Split-Path -Parent -Path $PSScriptRoot
    $fileDialog.InitialDirectory = $scriptPath
    
    $dialogResult = $fileDialog.ShowDialog()

    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        return $fileDialog.FileName
    }
    else {
        return $null
    }
}

# Function to select folder
function Select-Folder([string]$description) {
    Write-Host $description -ForegroundColor Green
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.Description = $description
    $folderBrowser.SelectedPath = [System.Environment]::GetFolderPath('MyDocuments')
    $dialogResult = $folderBrowser.ShowDialog()

    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        return $folderBrowser.SelectedPath
    }
    else {
        return $null
    }
}

# Function to select output extension
function Select-OutputExtension {
    $options = @(".holo", ".mp4", ".avi")
    Write-Host "Select the output file extension:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $options.Length; $i++) {
        Write-Host "$($i+1). $($options[$i])" -ForegroundColor Yellow
    }
    $selection = Read-Host "Enter the number corresponding to your choice (default is 3 for .avi)"

    if ([string]::IsNullOrEmpty($selection) -or $selection -lt 1 -or $selection -gt $options.Length) {
        return ".avi"
    }
    else {
        return $options[$selection - 1]
    }
}

# Function to get configuration files
function Get-ConfigFileOption {
    $filePath = Select-File -description "Select a configuration file" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*"
    if ($filePath) {
        Write-Host "Selected configuration file: $filePath" -ForegroundColor Green
    }
    return $filePath
}

# Prompt the user to select the folder containing .holo files
$holoFolderPath = Select-Folder -description "Select the folder containing .holo files"
if (-not $holoFolderPath) {
    Write-Host "No folder was selected. Exiting script." -ForegroundColor Red
    exit
}

# Select recording mode
Write-Host "Select recording mode:" -ForegroundColor Cyan
Write-Host "1. Normal recording" -ForegroundColor Yellow
Write-Host "2. Moments recording" -ForegroundColor Yellow
Write-Host "3. Both (Normal + Moments with different configurations)" -ForegroundColor Yellow
$modeChoice = Read-Host "Enter the number corresponding to your choice (default is 1)"

if (-not ($modeChoice -match '^[1-3]$')) { $modeChoice = 1 }

$configFileNormal = $null
$configFileMoments = $null

# Get configuration files based on mode
switch ($modeChoice) {
    1 { $configFileNormal = Get-ConfigFileOption }
    2 { $configFileMoments = Get-ConfigFileOption }
    3 {
        Write-Host "Select configuration file for NORMAL recording:" -ForegroundColor Cyan
        $configFileNormal = Get-ConfigFileOption
        Write-Host "Select configuration file for MOMENTS recording:" -ForegroundColor Cyan
        $configFileMoments = Get-ConfigFileOption
    }
}

# Set the frame skip and input fps
$frameSkip = Read-Host "Enter frame skip (default 16)"
if (-not ($frameSkip -match '^[0-9]+$')) { $frameSkip = 16 }

$input_fps = Read-Host "Enter input fps (optional, default camera fps)"
if (-not ($input_fps -match '^[0-9]+$')) { $input_fps = -1 }

$outputExtension = Select-OutputExtension

# List all .holo files in the selected folder
$holoFiles = Get-ChildItem -Path $holoFolderPath -Filter *.holo -Recurse
if (-not $holoFiles) {
    Write-Host "No .holo files found in the selected folder. Exiting script." -ForegroundColor Red
    exit
}

Write-Host "Found $($holoFiles.Count) .holo files in the selected folder." -ForegroundColor Cyan

# Find the holovibes .exe
$exePath1 = "Holovibes.exe"
$exePath2 = "build/bin/Holovibes.exe"
$exePath = ""

# Check if Holovibes.exe exists
if (Test-Path $exePath1) {
    $exePath = $exePath1
} else {
    $exePath = $exePath2
}

Write-Host "Using Holovibes executable: $exePath" -ForegroundColor Cyan

# Function to execute Holovibes
function Execute-Holovibes {
    param (
        [string]$inputFilePath,
        [string]$outputFilePath,
        [string]$frameSkip,
        [string]$input_fps,
        [string]$configFile,
        [bool]$isMoments
    )

    $args = "-i `"$inputFilePath`" -o `"$outputFilePath`""
    if ($input_fps -ne -1) { $args += " -f $input_fps" }
    if ($isMoments) {
        $args += " --moments_record"
    } else {
        $args += " --frame_skip $frameSkip"
    }
    if ($configFile) { $args += " -c `"$configFile`"" }

    Write-Host "Executing Holovibes with arguments: $args" -ForegroundColor Yellow
    Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -Wait
}

# Process files
foreach ($file in $holoFiles) {
    $inputFilePath = $file.FullName

    if ($modeChoice -eq 1 -or $modeChoice -eq 3) {
        $outputFilePath = Join-Path $holoFolderPath "$($file.BaseName)_normal$outputExtension"
        Execute-Holovibes -inputFilePath $inputFilePath -outputFilePath $outputFilePath -frameSkip $frameSkip -input_fps $input_fps -configFile $configFileNormal -isMoments $false
    }

    if ($modeChoice -eq 2 -or $modeChoice -eq 3) {
        $outputFilePath = Join-Path $holoFolderPath "$($file.BaseName)_moments.holo"
        Execute-Holovibes -inputFilePath $inputFilePath -outputFilePath $outputFilePath -frameSkip $frameSkip -input_fps $input_fps -configFile $configFileMoments -isMoments $true
    }
}

Write-Host "Processing complete. All files have been processed." -ForegroundColor Cyan
