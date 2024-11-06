# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms

# Function to prompt the user to select a file, starting at the last used folder
function Select-File([string]$description, [string]$filter) {
    Write-Host $description -ForegroundColor Green
    $fileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $fileDialog.Filter = $filter
    $fileDialog.Title = $description
    $fileDialog.InitialDirectory = [System.Environment]::GetFolderPath('MyDocuments')
    $dialogResult = $fileDialog.ShowDialog()

    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK) {
        return $fileDialog.FileName
    }
    else {
        return $null
    }
}


# Function to prompt the user to select a folder, starting at the last used folder
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



# Prompt the user to select the folder containing .holo files
$holoFolderPath = Select-Folder -description "Select the folder containing .holo files"

# Check if the user selected a folder
if (-not $holoFolderPath) {
    Write-Host "No folder was selected. Exiting script." -ForegroundColor Red
    exit
}

$configFile1 = "AppData/preset/registration.json"
$configFile2 = "Preset/registration.json"
$configFiles = ""

# Check if ../Holovibes.exe exists
if (Test-Path $configFile1) {
    $configFiles = $configFile1
}
else {
    $configFiles = $configFile2
}

$exePath1 = "Holovibes.exe"
$exePath2 = "build/bin/Holovibes.exe"
$exePath = ""

# Check if ../Holovibes.exe exists
if (Test-Path $exePath1) {
    $exePath = $exePath1
}
else {
    $exePath = $exePath2
}
# Set the frame skip to 16
$frameSkip = 16
#$frameSkip = Read-Host -Prompt "Enter the frame skip you want (optional)"
#if (-not ($frameSkip -match '^\d+$')) {
#    $frameSkip = "0"
#}


# Confirm action with the user
Write-Host "You have selected the folder: $holoFolderPath" -ForegroundColor Cyan
Write-Host "Using Holovibes executable: $exePath" -ForegroundColor Cyan

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
    }
    else {
        return $options[$selection - 1]
    }
}


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

# Get the total number of operations
$total = 0

$total = ($counter - 1)

$counter = 1
$number_pb = 0

# Confirm action with the user before processing files
Read-Host "Press Enter to start processing files or Ctrl+C to exit"

# Function to run Holovibes in CLI
function Execute-Holovibes {
    param (
        [string]$inputFilePath,
        [string]$outputFilePath,
        [string]$frameSkip,
        [string]$configFile = $null
    )

    $args = "-i `"$inputFilePath`" -o `"$outputFilePath`""
    $args += " -c `"$configFile`""
    $args += " --frame_skip 8"
    Write-Host "Processing $(Split-Path -Leaf $inputFilePath) with config $configFile..." -ForegroundColor Yellow
   
    $args
    Write-Host "Processing holo files ($counter/$total)" -ForegroundColor Cyan
    $process = (Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -PassThru -Wait)
    if ($process.ExitCode -ne 0) {
        Write-Host "Program exited with bad exit code" -ForegroundColor Red
        $script:number_pb += 1
    }
    else {
        Write-Host "Finished processing $(Split-Path -Leaf $inputFilePath), output saved as $(Split-Path -Leaf $outputFilePath)" -ForegroundColor Green
    }
}

# Execute Holovibes.exe for each .holo file and each configuration file (or no config file)
foreach ($file in $holoFiles) {
    $inputFilePath = $file.FullName
    $outputFileName = "$($file.BaseName)_out$outputExtension"
    $outputFilePath = Join-Path $holoFolderPath $outputFileName

    
    Execute-Holovibes -exePath $exePath -inputFilePath $inputFilePath -outputFilePath $outputFilePath -configFile $configFiles
    $counter += 1
    
}

# Final message after all files have been processed
Write-Host "All $total .holo files have been processed." -ForegroundColor Cyan
if ($number_pb -gt 0) {
    Write-Host "$number_pb returned an error." -ForegroundColor Red
}
