# VoxGen Windows scripts

All Windows build/smoke launchers live here. Run the root `build_voxgen.bat` for the normal release build.

The model directory defaults to `models\` in the project root when present, otherwise `C:\Software\VoxCPM-Q8\models`. Override with `VOXGEN_MODEL_DIR`, `VOXGEN_BASE_Q8`, `VOXGEN_BASE_F16`, or `VOXGEN_ACOUSTIC`.

Build modes:

```bat
build_voxgen.bat
build_voxgen.bat debug
build_voxgen.bat check
build_voxgen.bat clean
build_voxgen.bat release --no-probe
```
