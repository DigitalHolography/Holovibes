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
        @{ Name = 'lsci-pca-acc'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_window', '3', '--lsci_pca_image_acc', '3'); NoSettings = $true; Acc = 3 },
        @{ Name = 'lsci-pca-overlap'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_stride', '2', '--lsci_pca_window', '3', '--lsci_pca_image_acc', '3'); NoSettings = $true; Stride = 2; Starts = @(1, 3, 5); Acc = 3 },
        @{ Name = 'lsci-pca-gap'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '2', '--lsci_pca_stride', '3', '--lsci_pca_window', '3'); NoSettings = $true; Frames = 2; Stride = 3; Starts = @(1, 4, 7) },
        @{ Name = 'lsci-pca-range'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '3', '--lsci_pca_stride', '1', '--lsci_pca_window', '3', '-s', '2', '-e', '7', '-n', '2'); NoSettings = $true; Frames = 3; Stride = 1; Starts = @(2, 3) },
        @{ Name = 'lsci-pca-large-gap'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_stride', '100', '--lsci_pca_window', '3'); NoSettings = $true; Stride = 100; Starts = @(1) },
        @{ Name = 'lsci-pca-overlap-ref'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_window', '3', '-s', '3', '-e', '6'); NoSettings = $true; Starts = @(3) },
        @{ Name = 'lsci-pca-gap-ref'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '2', '--lsci_pca_window', '3', '-s', '4', '-e', '5'); NoSettings = $true; Frames = 2; Starts = @(4) },
        @{ Name = 'lsci-pca-stride-zero'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_stride', '0'); NoSettings = $true; Fails = $true },
        @{ Name = 'lsci-pca-stride-negative'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_stride', '-1'); NoSettings = $true; Fails = $true },
        @{ Name = 'lsci-pca-stride-count'; Extension = 'avi'; Options = @('--lsci_pca', '--lsci_pca_frames', '4', '--lsci_pca_stride', '2', '-n', '4'); NoSettings = $true; Fails = $true },
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
        if ($case.Fails) {
            if ($child.ExitCode -eq 0) { throw "CLI case $($case.Name) accepted invalid options. Logs: $outputDir" }
            continue
        }
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
        if ($case.Name -like 'lsci-pca*') {
            $metadata = Get-Content -LiteralPath "$outputFile.json" -Raw | ConvertFrom-Json
            $expectedFrames = if ($case.Frames) { $case.Frames } else { 4 }
            $expectedStride = if ($case.Stride) { $case.Stride } else { $expectedFrames }
            $expectedStarts = if ($case.ContainsKey('Starts')) { @($case.Starts) } else { @(1, 5) }
            if ($metadata.output_frames -ne $expectedStarts.Count -or $metadata.blocks.Count -ne $expectedStarts.Count -or
                -not $metadata.display_range_fixed -or $metadata.frames_per_block -ne $expectedFrames -or
                $metadata.time_stride -ne $expectedStride) {
                throw "PCA-LSCI block count or fixed display metadata is incorrect. Logs: $outputDir"
            }
            $expectedAcc = if ($case.Acc) { $case.Acc } else { 1 }
            if ($metadata.image_accumulation -ne $expectedAcc) {
                throw "PCA-LSCI image accumulation metadata is incorrect. Logs: $outputDir"
            }
            for ($block = 0; $block -lt $expectedStarts.Count; $block++) {
                if ($metadata.blocks[$block].input_first_frame -ne $expectedStarts[$block] -or
                    $metadata.blocks[$block].accumulated_maps -ne [Math]::Min($block + 1, $expectedAcc)) {
                    throw "PCA-LSCI selected the wrong input window or accumulation count. Logs: $outputDir"
                }
            }
        }
    }
    # Compare shifted windows with separately processed frame ranges, checking
    # that the PCA actually received those samples rather than just their labels.
    foreach ($name in @('overlap', 'gap')) {
        $actual = (Get-Content -LiteralPath "$outputDir/lsci-pca-$name.avi.json" -Raw | ConvertFrom-Json).blocks[1]
        $reference = (Get-Content -LiteralPath "$outputDir/lsci-pca-$name-ref.avi.json" -Raw | ConvertFrom-Json).blocks[0]
        if ($actual.static_components -ne $reference.static_components) { throw "PCA-LSCI $name input window mismatch." }
        for ($eigenvalue = 0; $eigenvalue -lt $reference.eigenvalues.Count; $eigenvalue++) {
            $scale = [Math]::Max(1.0, [Math]::Abs($reference.eigenvalues[$eigenvalue]))
            if ([Math]::Abs($actual.eigenvalues[$eigenvalue] - $reference.eigenvalues[$eigenvalue]) -gt 1e-5 * $scale) {
                throw "PCA-LSCI $name PCA differs from the separately processed input window. Logs: $outputDir"
            }
        }
    }
    Write-Host "CLI recordings and exports passed. Logs: $outputDir"
} finally {
    $env:APPDATA = $previousAppData
    $env:PATH = $previousPath
}
