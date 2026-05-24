use std::process::{Command, Output};
use std::time::Instant;

#[derive(Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub timed_out: bool,
}

pub struct Executor {
    pub python_cmd: String,
    pub timeout_ms: u64,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            python_cmd: "python3".to_string(),
            timeout_ms: 5000,
        }
    }

    pub fn with_python(cmd: &str) -> Self {
        Executor {
            python_cmd: cmd.to_string(),
            timeout_ms: 5000,
        }
    }

    pub fn run_python(&self, code: &str) -> ExecResult {
        let start = Instant::now();
        let mut child = match Command::new(&self.python_cmd)
            .arg("-c")
            .arg(code)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return ExecResult {
                    stdout: String::new(),
                    stderr: format!("Failed to spawn: {}", e),
                    exit_code: -1,
                    duration_ms: start.elapsed().as_millis() as u64,
                    timed_out: false,
                };
            }
        };

        let output: Output = loop {
            if start.elapsed().as_millis() as u64 >= self.timeout_ms {
                let _ = child.kill();
                let _ = child.wait();
                return ExecResult {
                    stdout: String::new(),
                    stderr: "[TIMEOUT] execution exceeded {}ms".to_string(),
                    exit_code: -1,
                    duration_ms: start.elapsed().as_millis() as u64,
                    timed_out: true,
                };
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    let out = child.wait_with_output().ok().unwrap_or_else(|| {
                        std::process::Output {
                            stdout: Vec::new(),
                            stderr: Vec::new(),
                            status: status,
                        }
                    });
                    break out;
                }
                Ok(None) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return ExecResult {
                        stdout: String::new(),
                        stderr: format!("Process error: {}", e),
                        exit_code: -1,
                        duration_ms: start.elapsed().as_millis() as u64,
                        timed_out: false,
                    };
                }
            }
        };

        let elapsed = start.elapsed().as_millis() as u64;
        ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration_ms: elapsed,
            timed_out: false,
        }
    }

    pub fn parse_output_to_tokens(text: &str) -> Vec<String> {
        let separators = [' ', '\t', '\n', '\r', '→', ',', ';', ':', '|'];
        let mut tokens: Vec<String> = text.split(&separators[..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() < 64)
            .collect();
        tokens.dedup();
        tokens.truncate(16);
        tokens
    }
}
