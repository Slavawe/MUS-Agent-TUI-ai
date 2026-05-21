pub mod ascii_tokenizer;
pub mod canvas;
pub mod renderer;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ascii_tokenizer::ASCIITokenizer>()?;
    m.add_class::<ascii_tokenizer::EncodeMode>()?;
    m.add_class::<canvas::CanvasLayer>()?;
    m.add_class::<canvas::VisionCanvas>()?;
    m.add_class::<renderer::VisionRenderer>()?;
    Ok(())
}
