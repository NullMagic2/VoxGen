param()

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path

function Say([string]$Message) {
    Write-Host "[VoxGen clean] $Message"
}

function Relative-To-Root([string]$Path) {
    $base = $Root.TrimEnd('\', '/')
    if ($Path.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
        return $Path.Substring($base.Length).TrimStart('\', '/')
    }
    return $Path
}

function Clean-TargetKeepBinary([string]$Target, [string[]]$Keep) {
    if (-not (Test-Path -LiteralPath $Target -PathType Container)) {
        return
    }

    $targetItem = Get-Item -LiteralPath $Target -Force
    if (($targetItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to clean target directory through a junction/symlink: $Target"
    }

    Say "Cleaning $(Relative-To-Root $Target)\ while preserving final binaries..."
    $base = $Target.TrimEnd('\', '/')

    Get-ChildItem -LiteralPath $Target -Force -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($base.Length + 1).Replace('/', '\')
        if ($Keep -notcontains $relative) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }

    # Remove file symlinks/reparse points that were not reported by -File.
    Get-ChildItem -LiteralPath $Target -Force -Recurse | Where-Object {
        -not $_.PSIsContainer -and
        (($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
    } | ForEach-Object {
        $relative = $_.FullName.Substring($base.Length + 1).Replace('/', '\')
        if ($Keep -notcontains $relative) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }

    # Bottom-up removal leaves release/debug in place only when they contain a
    # preserved executable.
    Get-ChildItem -LiteralPath $Target -Force -Recurse -Directory |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            if (($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                Remove-Item -LiteralPath $_.FullName -Force
            }
            elseif (-not (Get-ChildItem -LiteralPath $_.FullName -Force | Select-Object -First 1)) {
                Remove-Item -LiteralPath $_.FullName -Force
            }
        }
}

function Remove-LocalTree([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Say "Removing $(Relative-To-Root $Path)"
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        # Remove only the link/junction itself. Never recurse into an external
        # model/download/cache location.
        Remove-Item -LiteralPath $Path -Force
    }
    else {
        Remove-Item -LiteralPath $Path -Force -Recurse
    }
}

function Remove-LocalFile([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        Say "Removing $(Relative-To-Root $Path)"
        Remove-Item -LiteralPath $Path -Force
    }
}

Say "Project root: $Root"

Clean-TargetKeepBinary (Join-Path $Root 'target') @(
    'release\voxgen.exe',
    'debug\voxgen.exe',
    'release\voxgen',
    'debug\voxgen'
)

Clean-TargetKeepBinary (Join-Path $Root 'demo\target') @(
    'release\voxgen-demo.exe',
    'debug\voxgen-demo.exe',
    'release\voxgen-demo',
    'debug\voxgen-demo'
)

# Project-local downloads only. Do not touch the user's global Cargo cache or
# any model path outside this source tree.
@(
    'models',
    'downloads',
    '.cache',
    'demo\models',
    'demo\downloads',
    'demo\.cache'
) | ForEach-Object { Remove-LocalTree (Join-Path $Root $_) }

@(
    'Cargo.lock',
    'demo\Cargo.lock'
) | ForEach-Object { Remove-LocalFile (Join-Path $Root $_) }

# Outputs produced by the bundled smoke tests. Deterministic input fixtures
# shipped with the source remain untouched.
@(
    'test_cfm_output.f32',
    'test_conditioned_cfm_output.f32',
    'test_clone.wav',
    'test_continuation.wav',
    'test_expressive.wav',
    'test_tts.wav',
    'test_tts_stream.wav',
    'test_ultimate.wav',
    'test_vae_decode.wav',
    'test_vae_decode_pcm.f32',
    'test_vae_encoded.f32',
    'test_vae_pcm_encoded.f32',
    'test_vae_roundtrip.f32',
    'test_vae_roundtrip.wav'
) | ForEach-Object { Remove-LocalFile (Join-Path $Root $_) }

Say 'Done. Source/build/download artifacts were cleaned; final VoxGen and demo binaries were preserved when present.'
