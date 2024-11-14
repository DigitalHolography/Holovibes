# Import the necessary assembly for file dialog
Add-Type -AssemblyName System.Windows.Forms
Write-Host (Get-Location)
$moments=0
if ($args[0] -eq "-m")
{
    $moments=1
}

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

# Function to get a list of configuration files from the user
function Get-ConfigFiles {
    $configFiles = @()

    # Prompt for the first configuration file
    $configFilePath = Select-File -description "Select a configuration file (optional)" -filter "Config Files (*.json)|*.json|All Files (*.*)|*.*"

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
    }
    else {
        Write-Host "No configuration files selected." -ForegroundColor Yellow
    }

    return $configFiles
}


# Function to get directory name from full file path
function Get-DirectoryName([string]$filePath) {
    return [System.IO.Path]::GetDirectoryName($filePath)
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

# Get the list of configuration files from the user (only if the first one was selected)
$configFiles = Get-ConfigFiles

$exePath1 = "Holovibes.exe"
$exePath2 = "build/bin/Holovibes.exe"
$exePath = ""

# Check if Holovibes.exe exists
if (Test-Path $exePath1) {
    $exePath = $exePath1
} else {
    $exePath = $exePath2
}
# Set the frame skip to 16
$frameSkip = 16
$frameSkip = Read-Host -Prompt "Enter the frame skip you want (optional, default 16)"
if (-not ($frameSkip -match '^\d+$')) {
    $frameSkip = 16
}

# Confirm action with the user
Write-Host "You have selected the folder: $holoFolderPath" -ForegroundColor Cyan
if ($configFiles.Count -gt 0) {
    Write-Host "You have selected the following configuration files:" -ForegroundColor Cyan
    $configFiles | ForEach-Object { Write-Host " - $_" -ForegroundColor Cyan }
}
else {
    Write-Host "No configuration files selected, continuing without them." -ForegroundColor Yellow
}
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

$outputExtension = ".holo"
if ($moments -eq 0)
{
    # Prompt the user to select the output file extension
    $script:outputExtension = Select-OutputExtension
    Write-Host "Selected output extension: $outputExtension" -ForegroundColor Cyan
}

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
if ($configFiles.Count -gt 0)
{
    $total = $configFiles.Count * ($counter - 1)
}
else {
    $total = ($counter - 1)
}
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

    if ($moments -eq 1)
    {
        $args += " --moments_record"
    }
    else {
        if ($outputExtension -eq ".mp4")
        {
            $args += " --mp4_fps 24"
        } else {
            $args += " --frame_skip $frameSkip"
        }
    }

    if ($configFile) {
        $args += " -c `"$configFile`""
        Write-Host "Processing $(Split-Path -Leaf $inputFilePath) with config $configFile..." -ForegroundColor Yellow
    } else {
        Write-Host "Processing $(Split-Path -Leaf $inputFilePath) without configuration..." -ForegroundColor Yellow
    }
    $args
    Write-Host "Processing holo files ($counter/$total)" -ForegroundColor Cyan
    $process = (Start-Process -FilePath $exePath -ArgumentList $args -NoNewWindow -PassThru -Wait)
    if ($process.ExitCode -ne 0)
    {
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
    if ($moments -eq 1)
    {
        $outputFileName = "$($file.BaseName)_moments$outputExtension"
    }
    $outputFilePath = Join-Path $holoFolderPath $outputFileName

    if ($configFiles.Count -gt 0) {
        foreach ($configFile in $configFiles) {
            Execute-Holovibes -exePath $exePath -inputFilePath $inputFilePath -outputFilePath $outputFilePath -frameSkip $frameSkip -configFile $configFile
            $counter += 1
        }
    } else {
        Execute-Holovibes -exePath $exePath -inputFilePath $inputFilePath -outputFilePath $outputFilePath -frameSkip $frameSkip
        $counter += 1
    }
}

# Final message after all files have been processed
Write-Host "All $total .holo files have been processed." -ForegroundColor Cyan
if ($number_pb -gt 0)
{
    Write-Host "$number_pb returned an error." -ForegroundColor Red
}
