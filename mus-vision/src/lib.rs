pub mod cpp_tokenizer;
pub mod canvas;
pub mod renderer;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cpp_tokenizer::CPPTokenizer>()?;
    m.add_class::<cpp_tokenizer::EncodeMode>()?;
    m.add_class::<canvas::CanvasLayer>()?;
    m.add_class::<canvas::VisionCanvas>()?;
    m.add_class::<renderer::VisionRenderer>()?;
    Ok(())
}
