# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms

# Function to prompt the user to select a folder
function Select-Folder([string]$description) {
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.Description = $description
    $dialogResult = $folderBrowser.ShowDialog()
    
    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        return $folderBrowser.SelectedPath
    } else {
        return $null
    }
}

# Function to prompt the user to select a file
function Select-File([string]$description, [string]$filter) {
    $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $fileDialog.Filter = $filter
    $fileDialog.Title = $description
    $dialogResult = $fileDialog.ShowDialog()
    
    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        return $fileDialog.FileName
    } else {
        return $null
    }
}

# Function to prompt the user to select an output extension
function Select-OutputExtension {
    $options = @(".holo", ".mp4", ".avi")
    Write-Host "Select the output file extension:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $options.Length; $i++) {
        Write-Host "$($i+1). $($options[$i])" -ForegroundColor Yellow
    }
    $selection = Read-Host "Enter the number corresponding to your choice (default is 3 for .avi)"
    
    if ([string]::IsNullOrEmpty($selection) -or $selection -lt 1 -or $selection -gt $options.Length) {
        return ".avi"
    } else {
        return $options[$selection - 1]
    }
}

# Prompt the user to select the folder containing .holo files
$holoFolderPath = Select-Folder -description "Select the folder containing .holo files"

# Check if the user selected a folder
if (-not $holoFolderPath) {
    Write-Host "No folder was selected. Exiting script." -ForegroundColor Red
    exit
}

# Prompt the user to select the configuration file (optional)
$configFilePath = Select-File -description "Select the configuration file (optional)" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*"

# Prompt the user to select the Holovibes executable (optional)
$exePath = Select-File -description "Select the Holovibes executable (optional)" -filter "Executable Files (*.exe)|*.exe|All Files (*.*)|*.*"

# Check if the user selected the executable, if not, use local or PATH version
if (-not $exePath) {
    $localExePath = Join-Path -Path (Get-Location) -ChildPath "Holovibes.exe"
    if (Test-Path $localExePath) {
        $exePath = $localExePath
    } else {
        $exePath = "Holovibes.exe"  # This assumes it's available in PATH
    }
}

# Confirm action with the user
Write-Host "You have selected the folder: $holoFolderPath" -ForegroundColor Cyan
if ($configFilePath) {
    Write-Host "You have selected the configuration file: $configFilePath" -ForegroundColor Cyan
} else {
    Write-Host "No configuration file selected, continuing without it." -ForegroundColor Yellow
}
Write-Host "Using Holovibes executable: $exePath" -ForegroundColor Cyan
Read-Host "Press Enter to continue or Ctrl+C to exit"

# Prompt the user to select the output file extension
$outputExtension = Select-OutputExtension
Write-Host "Selected output extension: $outputExtension" -ForegroundColor Cyan

# Get a list of all .holo files in the selected folder
$holoFiles = Get-ChildItem -Path $holoFolderPath -Filter *.holo

# Display the list of .holo files with colorful output
Write-Host "Listing .holo files in the selected folder:" -ForegroundColor Cyan
$counter = 1
foreach ($file in $holoFiles) {
    Write-Host "$counter. $($file.Name)" -ForegroundColor Green
    $counter++
}

# Confirm action with the user before processing files
Read-Host "Press Enter to start processing files or Ctrl+C to exit"

# Execute Holovibes.exe for each .holo file
foreach ($file in $holoFiles) {
    $inputFilePath = $file.FullName
    $outputFileName = "$($file.BaseName)_out$outputExtension"
    $outputFilePath = Join-Path $holoFolderPath $outputFileName

    # Prepare arguments for Holovibes.exe
    $args = "-i `"$inputFilePath`" -o `"$outputFilePath`""
    if ($configFilePath) {
        $args += " -c `"$configFilePath`""
    }

    # Run Holovibes.exe with the .holo file, specifying the output file name and config (if provided)
    Write-Host "Processing $($file.Name)..." -ForegroundColor Yellow
    Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -Wait
    Write-Host "Finished processing $($file.Name), output saved as $outputFileName" -ForegroundColor Green
}

Write-Host "All files processed successfully." -ForegroundColor Cyan
Read-Host "Press Enter to exit."
