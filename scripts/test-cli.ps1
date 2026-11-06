param([Parameter(Mandatory = $true)][string]$Executable)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$inputFile = Join-Path $repoRoot 'tests/data/inputs/8frames_64_64.holo'
$settings = Join-Path $repoRoot 'tests/data/8frames_64_64_space_1fft_time_stft_no_args/holovibes.json'
$inputBytes = [IO.File]::ReadAllBytes($inputFile)
if ($inputBytes.Length -ne 65600) { throw 'The CLI fixture is missing or still a Git LFS pointer. Run git lfs pull.' }
$expectedPixels = [Convert]::ToBase64String($inputBytes, 64, $inputBytes.Length - 64)
$outputDir = Join-Path $repoRoot "out/cli-smoke/$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force "$outputDir/appdata" | Out-Null
$previousAppData = $env:APPDATA
$previousPath = $env:PATH
try {
    $env:APPDATA = Join-Path $outputDir 'appdata'
    # Verify deployment without the CUDA Toolkit or vcpkg directories on PATH.
    $env:PATH = "$env:SystemRoot;$env:SystemRoot\System32"
    $cases = @(1..10 | ForEach-Object { @{ Name = "raw-$_"; Extension = 'holo'; Options = @('--raw', '-n', '8') } })
    $cases += @(
        @{ Name = 'processed'; Extension = 'holo'; Options = @() },
        @{ Name = 'avi'; Extension = 'avi'; Options = @('--fps', '240') },
        @{ Name = 'mp4'; Extension = 'mp4'; Options = @('--fps', '240') },
        @{ Name = 'lsci-pca'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_window', '3'); NoSettings = $true },
        @{ Name = 'lsci-pca-acc'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_window', '3', '--lsci_pca_image_acc', '3'); NoSettings = $true },
        @{ Name = 'moments'; Extension = 'h5'; Options = @('--moments_record') }
    )
    foreach ($case in $cases) {
        $outputFile = Join-Path $outputDir "$($case.Name).$($case.Extension)"
        $arguments = @('-i', "`"$inputFile`"", '-o', "`"$outputFile`"")
        if (-not $case.NoSettings) { $arguments += @('-c', "`"$settings`"") }
        $arguments += $case.Options
        $child = Start-Process -FilePath $Executable -ArgumentList $arguments -WindowStyle Hidden -PassThru `
            -WorkingDirectory (Split-Path -Parent $Executable) `
            -RedirectStandardOutput "$outputDir/$($case.Name).stdout.log" `
            -RedirectStandardError "$outputDir/$($case.Name).stderr.log"
        # Keep the process handle so Windows PowerShell retains ExitCode even
        # when this small recording finishes before WaitForExit is called.
        $null = $child.Handle
        if (-not $child.WaitForExit(60000)) {
            $child.Kill()
            throw "CLI case $($case.Name) timed out. Logs: $outputDir"
        }
        $child.WaitForExit()
        if ($child.ExitCode -ne 0) { throw "CLI case $($case.Name) failed ($($child.ExitCode)). Logs: $outputDir" }
        $outputs = @(Get-ChildItem -LiteralPath $outputDir -File -Filter "*$($case.Name).$($case.Extension)")
        if ($outputs.Count -eq 0 -or ($outputs | Where-Object Length -EQ 0)) {
            throw "CLI case $($case.Name) produced no output. Logs: $outputDir"
        }
        if ($case.Name -like 'raw-*') {
            $actual = [IO.File]::ReadAllBytes($outputs[0].FullName)
            if ($actual.Length -lt $inputBytes.Length -or
                [Convert]::ToBase64String($actual, 64, $inputBytes.Length - 64) -ne $expectedPixels) {
                throw "Raw pixel data changed in $($case.Name). Logs: $outputDir"
            }
        }
        if ($case.Name -in 'lsci-pca', 'lsci-pca-acc') {
            $metadata = Get-Content -LiteralPath "$outputFile.json" -Raw | ConvertFrom-Json
            if ($metadata.output_frames -ne 2 -or $metadata.blocks.Count -ne 2 -or
                -not $metadata.display_range_fixed -or $metadata.frames_per_block -ne 4) {
                throw "PCA-LSCI block count or fixed display metadata is incorrect. Logs: $outputDir"
            }
            $expectedAcc = if ($case.Name -eq 'lsci-pca-acc') { 3 } else { 1 }
            if ($metadata.image_accumulation -ne $expectedAcc -or
                $metadata.blocks[1].accumulated_maps -ne [Math]::Min(2, $expectedAcc)) {
                throw "PCA-LSCI image accumulation metadata is incorrect. Logs: $outputDir"
            }
        }
    }
    Write-Host "CLI recordings and exports passed. Logs: $outputDir"
} finally {
    $env:APPDATA = $previousAppData
    $env:PATH = $previousPath
}
