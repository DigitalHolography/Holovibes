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

# Prompt the user to select the folder containing .holo files
$holoFolderPath = Select-Folder -description "Select the folder containing .holo files"

# Check if the user selected a folder
if (-not $holoFolderPath) {
    Write-Host "No folder was selected. Exiting script." -ForegroundColor Red
    exit
}

# Prompt the user to select the configuration file
$configFilePath = Select-File -description "Select the configuration file" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*"

# Check if the user selected a config file
if (-not $configFilePath) {
    Write-Host "No configuration file was selected. Exiting script." -ForegroundColor Red
    exit
}

# Prompt the user to select the Holovibes executable
$exePath = Select-File -description "Select the Holovibes executable" -filter "Executable Files (*.exe)|*.exe|All Files (*.*)|*.*"

# Check if the user selected the executable
if (-not $exePath) {
    Write-Host "No executable was selected. Exiting script." -ForegroundColor Red
    exit
}

# Confirm action with the user
Write-Host "You have selected the folder: $holoFolderPath" -ForegroundColor Cyan
Write-Host "You have selected the configuration file: $configFilePath" -ForegroundColor Cyan
Write-Host "You have selected the Holovibes executable: $exePath" -ForegroundColor Cyan
Read-Host "Press Enter to continue or Ctrl+C to exit"

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
    $outputFileName = "$($file.BaseName)_out.mp4"
    $outputFilePath = Join-Path $holoFolderPath $outputFileName

    # Run Holovibes.exe with the .holo file, specifying the output file name and config
    Write-Host "Processing $($file.Name)..." -ForegroundColor Yellow
    Start-Process -FilePath $exePath -ArgumentList "-i `"$inputFilePath`" -o `"$outputFilePath`" -c `"$configFilePath`"" -NoNewWindow -Wait
    Write-Host "Finished processing $($file.Name), output saved as $outputFileName" -ForegroundColor Green
}

Write-Host "All files processed successfully." -ForegroundColor Cyan
Read-Host "Press Enter to exit."