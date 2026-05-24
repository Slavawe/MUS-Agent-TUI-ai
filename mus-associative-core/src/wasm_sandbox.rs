use wasmtime::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SandboxOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

pub struct WasmSandbox {
    engine: Engine,
}

impl WasmSandbox {
    pub fn new() -> Result<Self, anyhow::Error> {
        let mut config = Config::default();
        config.consume_fuel(true);
        config.max_wasm_stack(512 * 1024);
        let engine = Engine::new(&config)?;
        Ok(WasmSandbox { engine })
    }

    pub fn run_file(&self, wasm_path: &str, input: &str) -> Result<SandboxOutput, anyhow::Error> {
        let start = Instant::now();
        let module = Module::from_file(&self.engine, wasm_path)?;
        self.run_module(&module, input, start)
    }

    pub fn run_bytes(&self, wasm_bytes: &[u8], input: &str) -> Result<SandboxOutput, anyhow::Error> {
        let start = Instant::now();
        let module = Module::new(&self.engine, wasm_bytes)?;
        self.run_module(&module, input, start)
    }

    fn run_module(
        &self,
        module: &Module,
        input: &str,
        start: Instant,
    ) -> Result<SandboxOutput, anyhow::Error> {
        let mut store = Store::new(&self.engine, ());
        store.set_fuel(500_000)?;

        let instance = Instance::new(&mut store, module, &[])?;

        // Extract memory reference and funcs before borrowing store
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("WASM must export 'memory'"))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| anyhow::anyhow!("WASM must export 'alloc(i32)->i32'"))?;
        let run = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, "run")
            .map_err(|_| anyhow::anyhow!("WASM must export 'run(i32,i32)->i32'"))?;

        // Write input into WASM memory via the allocator
        let input_bytes = input.as_bytes();
        let input_len = input_bytes.len() as i32;
        let input_ptr = alloc.call(&mut store, input_len)?;
        memory.write(&mut store, input_ptr as usize, input_bytes)?;

        // Execute
        let result_ptr = run.call(&mut store, (input_ptr, input_len))?;

        // Read length-prefixed result
        let mut len_buf = [0u8; 4];
        memory.read(&mut store, result_ptr as usize, &mut len_buf)?;
        let out_len = i32::from_le_bytes(len_buf) as usize;
        let mut out_buf = vec![0u8; out_len];
        memory.read(&mut store, (result_ptr + 4) as usize, &mut out_buf)?;
        let output = String::from_utf8_lossy(&out_buf).to_string();

        let fuel = store.get_fuel()?;
        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(SandboxOutput {
            exit_code: if fuel > 0 { 0 } else { 1 },
            stdout: output,
            stderr: String::new(),
            duration_ms,
        })
    }
}
