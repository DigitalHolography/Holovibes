# Parse the small wrapper CLI explicitly: Windows PowerShell's -File binder
# treats a forwarded `--` as an ambiguous parameter on advanced scripts.
$ErrorActionPreference = 'Stop'
try {
    $Action = 'build'
    $Preset = 'windows-dev'
    $ExtraArgs = @()
    $wrapperArgs = @($args)
    for ($index = 0; $index -lt $wrapperArgs.Count; $index++) {
        $argument = $wrapperArgs[$index]
        if ($argument -eq '--') {
            $ExtraArgs = @($wrapperArgs | Select-Object -Skip ($index + 1))
            break
        }
        if ($argument -in @('-Preset', '-Action')) {
            if (++$index -ge $wrapperArgs.Count) { throw "Missing value for $argument." }
            if ($argument -eq '-Preset') { $Preset = $wrapperArgs[$index] }
            else { $Action = $wrapperArgs[$index] }
        } elseif ($index -eq 0 -and $argument -notlike '-*') {
            $Action = $argument
        } else {
            throw "Unknown wrapper argument: $argument. Put command arguments after --."
        }
    }
    if ($Action -notin @('setup', 'configure', 'build', 'run', 'test', 'package', 'clean', 'dependencies')) {
        throw "Unknown action: $Action. Use setup, configure, build, run, test, package, clean, or dependencies."
    }
    if ($Preset -notin @('windows-dev', 'windows-debug', 'windows-release', 'windows-tests', 'windows-dependencies')) {
        throw "Unknown preset: $Preset. Use a preset from CMakePresets.json."
    }
    if ($Action -eq 'dependencies') { $Preset = 'windows-dependencies' }
    if ($Action -eq 'test' -and $Preset -eq 'windows-dev') { $Preset = 'windows-tests' }
    if ($Action -eq 'package' -and $Preset -eq 'windows-dev') { $Preset = 'windows-release' }
    $buildPath = Join-Path $PSScriptRoot "out/build/$Preset"

    if ($Action -eq 'clean') {
        # Only this preset's known output directory can be removed.
        $buildRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'out/build'))
        $resolvedBuild = [IO.Path]::GetFullPath($buildPath)
        if (-not $resolvedBuild.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing to remove a path outside out/build.'
        }
        if (Test-Path -LiteralPath $resolvedBuild) { Remove-Item -LiteralPath $resolvedBuild -Recurse -Force }
        exit 0
    }

    . (Join-Path $PSScriptRoot 'scripts/bootstrap.ps1') -DependenciesOnly:($Preset -eq 'windows-dependencies')
    Push-Location $PSScriptRoot
    try {
        switch ($Action) {
            'setup' { }
            'configure' { Invoke-Checked cmake (@('--preset', $Preset) + $ExtraArgs) }
            'run' {
                $executable = Join-Path $buildPath 'Holovibes.exe'
                if (-not (Test-Path -LiteralPath $executable)) { throw "Build $Preset before running it." }
                Push-Location $buildPath
                try { Invoke-Checked $executable $ExtraArgs } finally { Pop-Location }
            }
            default {
                # Preserve explicit -D overrides from `configure`. Ninja asks
                # CMake to regenerate when a CMake input changes afterward.
                # A failed configure can leave a cache without generating Ninja files.
                if (-not (Test-Path -LiteralPath (Join-Path $buildPath 'build.ninja'))) {
                    Invoke-Checked cmake @('--preset', $Preset)
                }
                if ($Action -eq 'build') {
                    Invoke-Checked cmake (@('--build', '--preset', $Preset) + $ExtraArgs)
                } else {
                    Invoke-Checked cmake @('--build', '--preset', $Preset)
                    if ($Action -in @('test', 'dependencies')) {
                        if ($Preset -notin @('windows-tests', 'windows-dependencies')) {
                            throw 'Use windows-tests or windows-dependencies to run tests.'
                        }
                        Invoke-Checked ctest (@('--preset', $Preset) + $ExtraArgs)
                    }
                    if ($Action -eq 'package') {
                        if ($Preset -ne 'windows-release') { throw 'Packaging uses the windows-release preset.' }
                        Invoke-Checked cpack (@('--preset', 'windows-release') + $ExtraArgs)
                    }
                }
            }
        }
    } finally { Pop-Location }
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
