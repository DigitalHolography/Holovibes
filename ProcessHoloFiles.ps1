# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms

# Function to prompt the user to select a file, starting at the last used folder
function Select-File([string]$description, [string]$filter, [string]$envVarName) {
    Write-Host $description -ForegroundColor Green
    $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $fileDialog.Filter = $filter
    $fileDialog.Title = $description
    $fileDialog.InitialDirectory = Get-LastPath $envVarName ([System.Environment]::GetFolderPath('MyDocuments'))
    $dialogResult = $fileDialog.ShowDialog()

    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        Save-LastPath $envVarName (Get-DirectoryName $fileDialog.FileName)
        return $fileDialog.FileName
    } else {
        return $null
    }
}

# Function to get a list of configuration files from the user
function Get-ConfigFiles {
    $configFiles = @()

    # Prompt for the first configuration file
    $configFilePath = Select-File -description "Select a configuration file (optional)" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*" -envVarName $configFileEnvVar

    if ($configFilePath) {
        # Add the first configuration file
        $configFiles += $configFilePath
        Write-Host "Added configuration file: $configFilePath" -ForegroundColor Green

        # Ask if the user wants to add more files
        do {
            $addMore = Read-Host "Do you want to add another configuration file? (Y/N)"
            if ($addMore -eq 'Y' -or $addMore -eq 'y') {
                $configFilePath = Select-File -description "Select another configuration file (optional)" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*" -envVarName $configFileEnvVar
                if ($configFilePath) {
                    $configFiles += $configFilePath
                    Write-Host "Added configuration file: $configFilePath" -ForegroundColor Green
                }
            }
        } while ($addMore -eq 'Y' -or $addMore -eq 'y')
    } else {
        Write-Host "No configuration files selected." -ForegroundColor Yellow
    }

    return $configFiles
}

# Function to get the last used path from an environment variable
function Get-LastPath([string]$envVarName, [string]$defaultFolder) {
    $lastPath = [System.Environment]::GetEnvironmentVariable($envVarName, [System.EnvironmentVariableTarget]::User)
    if ($lastPath -and (Test-Path $lastPath)) {
        return $lastPath
    } else {
        return $defaultFolder
    }
}

# Function to save the last used path in an environment variable
function Save-LastPath([string]$envVarName, [string]$path) {
    if (![string]::IsNullOrEmpty($envVarName) -and (Test-Path $path)) {
        [System.Environment]::SetEnvironmentVariable($envVarName, $path, [System.EnvironmentVariableTarget]::User)
    } else {
        Write-Host "Invalid variable name or path: $envVarName, $path" -ForegroundColor Red
    }
}

# Function to get directory name from full file path
function Get-DirectoryName([string]$filePath) {
    return [System.IO.Path]::GetDirectoryName($filePath)
}

# Function to prompt the user to select a folder, starting at the last used folder
function Select-Folder([string]$description, [string]$envVarName) {
    Write-Host $description -ForegroundColor Green
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.Description = $description
    $folderBrowser.SelectedPath = Get-LastPath $envVarName ([System.Environment]::GetFolderPath('MyDocuments'))
    $dialogResult = $folderBrowser.ShowDialog()

    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        Save-LastPath $envVarName $folderBrowser.SelectedPath
        return $folderBrowser.SelectedPath
    } else {
        return $null
    }
}

# Names of the environment variables to store paths
$holoFolderEnvVar = "LAST_HOLO_FOLDER"
$exeFileEnvVar = "LAST_EXE_FILE"
$configFileEnvVar = "LAST_CONFIG_PATH"

# Prompt the user to select the folder containing .holo files
$holoFolderPath = Select-Folder -description "Select the folder containing .holo files" -envVarName $holoFolderEnvVar

# Check if the user selected a folder
if (-not $holoFolderPath) {
    Write-Host "No folder was selected. Exiting script." -ForegroundColor Red
    exit
}

# Get the list of configuration files from the user (only if the first one was selected)
$configFiles = Get-ConfigFiles

# Prompt the user to select the Holovibes executable (optional)
$exePath = Select-File -description "Select the Holovibes executable (optional)" -filter "Executable Files (*.exe)|*.exe|All Files (*.*)|*.*" -envVarName $exeFileEnvVar

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
if ($configFiles.Count -gt 0) {
    Write-Host "You have selected the following configuration files:" -ForegroundColor Cyan
    $configFiles | ForEach-Object { Write-Host " - $_" -ForegroundColor Cyan }
} else {
    Write-Host "No configuration files selected, continuing without them." -ForegroundColor Yellow
}
Write-Host "Using Holovibes executable: $exePath" -ForegroundColor Cyan
Read-Host "Press Enter to continue or Ctrl+C to exit"

# Function to prompt the user to select the output file extension
function Select-OutputExtension {
    Write-Host "Please select the desired output file extension:" -ForegroundColor Green
    $extensions = @(".jpg", ".png", ".bmp")
    $counter = 1
    foreach ($ext in $extensions) {
        Write-Host "$counter. $ext" -ForegroundColor Cyan
        $counter++
    }
    $selection = Read-Host "Enter the number of your choice"

    switch ($selection) {
        1 { return ".jpg" }
        2 { return ".png" }
        3 { return ".bmp" }
        default { Write-Host "Invalid selection, defaulting to .jpg" -ForegroundColor Yellow; return ".jpg" }
    }
}

# Prompt the user to select the output file extension
$outputExtension = Select-OutputExtension
Write-Host "Selected output extension: $outputExtension" -ForegroundColor Cyan

# Get a list of all .holo files in the selected folder
$holoFiles = Get-ChildItem -Path $holoFolderPath -Filter *.holo -Recurse

# Display the list of .holo files with colorful output
Write-Host "Listing .holo files in the selected folder:" -ForegroundColor Cyan
$counter = 1
foreach ($file in $holoFiles) {
    Write-Host "$counter. $($file.Name)" -ForegroundColor Green
    $counter++
}

# Confirm action with the user before processing files
Read-Host "Press Enter to start processing files or Ctrl+C to exit"

# Execute Holovibes.exe for each .holo file and each configuration file (or no config file)
foreach ($file in $holoFiles) {
    if ($configFiles.Count -gt 0) {
        foreach ($configFile in $configFiles) {
            $inputFilePath = $file.FullName
            $outputFileName = "$($file.BaseName)_out$outputExtension"
            $outputFilePath = Join-Path $holoFolderPath $outputFileName

            # Prepare arguments for Holovibes.exe
            $args = "-i `"$inputFilePath`" -o `"$outputFilePath`" -c `"$configFile`""

            # Run Holovibes.exe with the .holo file and the current configuration file
            Write-Host "Processing $($file.Name) with config $($configFile)..." -ForegroundColor Yellow
            Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -Wait
            Write-Host "Finished processing $($file.Name) with config $($configFile), output saved as $outputFileName" -ForegroundColor Green
        }
    } else {
        $inputFilePath = $file.FullName
        $outputFileName = "$($file.BaseName)_out$outputExtension"
        $outputFilePath = Join-Path $holoFolderPath $outputFileName

        # Prepare arguments for Holovibes.exe without configuration file
        $args = "-i `"$inputFilePath`" -o `"$outputFilePath`""

        # Run Holovibes.exe with the .holo file and no configuration file
        Write-Host "Processing $($file.Name) without configuration..." -ForegroundColor Yellow
        Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -Wait
        Write-Host "Finished processing $($file.Name), output saved as $outputFileName" -ForegroundColor Green
    }
}

# Final message after all files have been processed
Write-Host "All .holo files have been processed." -ForegroundColor Cyan
