use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct LintResult {
    pub errors: u32,
    pub warnings: u32,
    pub notes: u32,
    pub duration_ms: u64,
    pub raw_output: String,
}

impl LintResult {
    /// Score: 1.0 = perfect, 0.0 = many errors
    pub fn score(&self) -> f32 {
        let penalty = (self.errors as f32 * 0.5) + (self.warnings as f32 * 0.15);
        (1.0 - penalty.min(1.0)).max(0.0)
    }

    pub fn is_clean(&self) -> bool {
        self.errors == 0 && self.warnings == 0
    }
}

pub struct Linter;

impl Linter {
    /// Run `cargo clippy` on a Rust project at the given path.
    /// Returns parsed lint results.
    pub fn clippy(project_path: &str) -> LintResult {
        let start = Instant::now();
        let output = Command::new("cargo")
            .args(["clippy", "--no-deps", "--message-format=short"])
            .current_dir(project_path)
            .output();

        let elapsed = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let combined = format!("{}\n{}", stdout, stderr);

                let errors = combined.lines()
                    .filter(|l| l.contains("error[") || l.starts_with("error:"))
                    .count() as u32;
                let warnings = combined.lines()
                    .filter(|l| l.contains("warning[") || l.starts_with("warning:"))
                    .count() as u32;
                let notes = combined.lines()
                    .filter(|l| l.contains("note:"))
                    .count() as u32;

                LintResult {
                    errors,
                    warnings,
                    notes,
                    duration_ms: elapsed,
                    raw_output: combined,
                }
            }
            Err(e) => LintResult {
                errors: 1,
                warnings: 0,
                notes: 0,
                duration_ms: elapsed,
                raw_output: format!("Failed to run clippy: {}", e),
            }
        }
    }

    /// Run `python3 -m py_compile` on a Python file for basic syntax check.
    pub fn py_compile(file_path: &str) -> LintResult {
        let start = Instant::now();
        let output = Command::new("python3")
            .args(["-m", "py_compile", file_path])
            .output();

        let elapsed = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let has_error = !stderr.is_empty() || !out.status.success();

                LintResult {
                    errors: if has_error { 1 } else { 0 },
                    warnings: 0,
                    notes: 0,
                    duration_ms: elapsed,
                    raw_output: stderr,
                }
            }
            Err(e) => LintResult {
                errors: 1,
                warnings: 0,
                notes: 0,
                duration_ms: elapsed,
                raw_output: format!("Failed to run py_compile: {}", e),
            }
        }
    }
}
