use pyo3::prelude::*;
use serde::Deserialize;

use crate::canvas::{CanvasLayer, VisionCanvas};

#[derive(Debug, Deserialize)]
struct YamlVisionMatrix {
    canvas_size: Option<String>,
    render_engine: Option<String>,
    layers: Option<Vec<YamlLayer>>,
}

#[derive(Debug, Deserialize)]
struct YamlLayer {
    id: Option<String>,
    #[serde(rename = "type")]
    layer_type: Option<String>,
    density: Option<f32>,
    coords: Option<Vec<i32>>,
    asset: Option<String>,
    bind_to: Option<String>,
}

#[pyclass]
pub struct VisionRenderer {
    canvas: VisionCanvas,
}

#[pymethods]
impl VisionRenderer {
    #[new]
    pub fn new(width: u16, height: u16) -> Self {
        VisionRenderer { canvas: VisionCanvas::new(width, height) }
    }

    #[pyo3(signature = (yaml_str, add_border=true))]
    pub fn render_yaml(&mut self, yaml_str: &str, add_border: bool) -> PyResult<(String, usize)> {
        self.canvas.clear_layers();

        let vm: YamlVisionMatrix = serde_yaml::from_str(yaml_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("YAML error: {}", e)))?;

        if let Some(ref size_str) = vm.canvas_size {
            if let Some((w, h)) = size_str.split_once('x') {
                if let (Ok(w), Ok(h)) = (w.trim().parse::<u16>(), h.trim().parse::<u16>()) {
                    self.canvas = VisionCanvas::new(w, h);
                }
            }
        }

        if add_border {
            self.canvas.add_border();
        }

        if let Some(ref layers) = vm.layers {
            for yl in layers {
                let layer_type = yl.layer_type.as_deref().unwrap_or("static").to_string();
                let (row, col) = if let Some(ref coords) = yl.coords {
                    (coords.first().copied().unwrap_or(0), coords.get(1).copied().unwrap_or(0))
                } else {
                    (0, 0)
                };
                let id = yl.id.as_deref().unwrap_or("unknown").to_string();
                let asset = yl.asset.as_deref().unwrap_or("").to_string();
                let density = yl.density.unwrap_or(0.2);
                self.canvas.add_layer(CanvasLayer::new(id, layer_type, asset, row, col, density));
            }
        }

        let ascii = self.canvas.render();
        let count = self.canvas.layer_count();
        Ok((ascii, count))
    }

    pub fn render_layers(&self) -> String {
        self.canvas.render()
    }

    pub fn add_layer(&mut self, id: String, layer_type: String, asset: String, row: i32, col: i32, density: f32) {
        self.canvas.add_layer(CanvasLayer::new(id, layer_type, asset, row, col, density));
    }

    pub fn add_border(&mut self) {
        self.canvas.add_border();
    }

    pub fn clear(&mut self) {
        self.canvas.clear_layers();
    }
}
