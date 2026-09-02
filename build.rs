use std::{env, fs, path::{Path, PathBuf}, process::Command};

fn glslc_path() -> PathBuf {
    if let Ok(p) = env::var("VOXGEN_GLSLC") {
        return PathBuf::from(p);
    }
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let root = PathBuf::from(sdk);
        let windows = root.join("Bin").join("glslc.exe");
        if windows.exists() { return windows; }
        let unix = root.join("bin").join("glslc");
        if unix.exists() { return unix; }
    }
    if cfg!(windows) { PathBuf::from("glslc.exe") } else { PathBuf::from("glslc") }
}

fn compile(glslc: &Path, src: &Path, dst: &Path) {
    let status = Command::new(glslc)
        .arg("--target-env=vulkan1.2")
        .arg("-O")
        .arg("-fshader-stage=compute")
        .arg(src)
        .arg("-o")
        .arg(dst)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to execute glslc at '{}': {e}. Install the Vulkan SDK or set VOXGEN_GLSLC to the glslc executable",
                glslc.display()
            )
        });
    if !status.success() {
        panic!("glslc failed while compiling {}", src.display());
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=VOXGEN_GLSLC");
    println!("cargo:rerun-if-changed=shaders");

    let glslc = glslc_path();
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let mut shaders: Vec<_> = fs::read_dir("shaders")
        .expect("read shaders directory")
        .filter_map(Result::ok)
        .map(|x| x.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("comp"))
        .collect();
    shaders.sort();

    for src in shaders {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let dst = out.join(format!("{stem}.spv"));
        println!("cargo:rerun-if-changed={}", src.display());
        compile(&glslc, &src, &dst);
    }
}
