use std::path::Path;
use std::process::Command;

fn find_cuda() -> String {
    for c in &["/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"] {
        let nvcc = format!("{}/bin/nvcc", c);
        if Path::new(&nvcc).exists() {
            return c.to_string();
        }
    }
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                let p = Path::new(&path);
                if let Some(parent) = p.parent().and_then(|p| p.parent()) {
                    return parent.to_str().unwrap().to_string();
                }
            }
        }
    }
    panic!("CUDA Toolkit not found. Install it or set PATH to nvcc.");
}

fn gpu_arch() -> String {
    if let Ok(arch) = std::env::var("MUS_CUDA_ARCH") {
        if !arch.is_empty() {
            return arch;
        }
    }
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Some((major, minor)) = cap.split_once('.') {
                let arch = format!("sm_{}{}", major, minor);
                eprintln!("  [build.rs] detected GPU arch: {}", arch);
                return arch;
            }
        }
    }
    eprintln!("  [build.rs] default GPU arch: sm_75");
    "sm_75".to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=src/native/graph_kernels.cu");
    println!("cargo:rerun-if-changed=src/native/bridge.cu");
    println!("cargo:rerun-if-changed=include/mus_associative.h");
    println!("cargo:rerun-if-env-changed=MUS_CUDA_ARCH");

    let arch = gpu_arch();
    let cuda = find_cuda();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let nvcc = format!("{}/bin/nvcc", cuda);
    let cuda_include = format!("{}/include", cuda);
    let cuda_lib = format!("{}/lib64", cuda);

    let srcs = vec![
        "src/native/graph_kernels.cu",
        "src/native/bridge.cu",
    ];

    let mut objs = Vec::new();
    for src in &srcs {
        let obj = format!("{}/{}.o", out_dir, src.replace('/', "_"));
        let status = Command::new(&nvcc)
            .arg("-c")
            .arg(&format!("-arch={}", arch))
            .arg("-O3")
            .arg("-use_fast_math")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-Iinclude")
            .arg(&format!("-I{}", cuda_include))
            .arg("-o")
            .arg(&obj)
            .arg(src)
            .status()
            .expect("nvcc failed");
        assert!(status.success(), "nvcc compile failed for {}", src);
        objs.push(obj);
    }

    // Device link
    let device_obj = format!("{}/device_link.o", out_dir);
    let mut dlink_cmd = Command::new(&nvcc);
    dlink_cmd.arg("-dlink")
        .arg(&format!("-arch={}", arch))
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-o").arg(&device_obj);
    for obj in &objs {
        dlink_cmd.arg(obj);
    }
    let status = dlink_cmd.status().expect("nvcc dlink failed");
    assert!(status.success(), "nvcc device link failed");
    objs.push(device_obj);

    // Static library
    let lib_path = format!("{}/libassoc_cuda.a", out_dir);
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &objs {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("ar failed");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=assoc_cuda");
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
