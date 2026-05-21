use pyo3::prelude::*;

const DENSITY_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '%', '#', '@'];

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct CanvasLayer {
    pub id: String,
    pub layer_type: String,
    pub asset: String,
    pub row: i32,
    pub col: i32,
    pub density: f32,
}

#[pymethods]
impl CanvasLayer {
    #[new]
    #[pyo3(signature = (id, layer_type="static".into(), asset="".into(), row=0, col=0, density=0.2))]
    pub fn new(id: String, layer_type: String, asset: String, row: i32, col: i32, density: f32) -> Self {
        CanvasLayer { id, layer_type, asset, row, col, density }
    }

    fn __repr__(&self) -> String {
        format!("CanvasLayer(id={}, type={}, pos=({},{}))", self.id, self.layer_type, self.row, self.col)
    }
}

#[pyclass]
pub struct VisionCanvas {
    pub width: u16,
    pub height: u16,
    layers: Vec<CanvasLayer>,
}

#[pymethods]
impl VisionCanvas {
    #[new]
    pub fn new(width: u16, height: u16) -> Self {
        VisionCanvas { width, height, layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: CanvasLayer) {
        self.layers.push(layer);
    }

    pub fn clear_layers(&mut self) {
        self.layers.clear();
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn render(&self) -> String {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut grid: Vec<Vec<char>> = vec![vec![' '; w]; h];

        for layer in &self.layers {
            match layer.layer_type.as_str() {
                "static" => Self::_render_static(&mut grid, layer),
                "sprite" => Self::_render_sprite(&mut grid, layer, self.width, self.height),
                _ => {}
            }
        }

        grid.iter().map(|row| row.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    pub fn add_border(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut border_asset = String::new();
        border_asset.push('+');
        for _ in 1..w.saturating_sub(1) { border_asset.push('-'); }
        if w > 1 { border_asset.push('+'); }
        border_asset.push('\n');
        for _ in 1..h.saturating_sub(1) {
            border_asset.push('|');
            for _ in 1..w.saturating_sub(1) { border_asset.push(' '); }
            if w > 1 { border_asset.push('|'); }
            border_asset.push('\n');
        }
        border_asset.push('+');
        for _ in 1..w.saturating_sub(1) { border_asset.push('-'); }
        if w > 1 { border_asset.push('+'); }

        self.layers.push(CanvasLayer {
            id: "border".into(),
            layer_type: "sprite".into(),
            asset: border_asset,
            row: 0, col: 0, density: 0.0,
        });
    }
}

impl VisionCanvas {
    fn _render_static(grid: &mut Vec<Vec<char>>, layer: &CanvasLayer) {
        let idx = ((layer.density.clamp(0.0, 1.0)) * (DENSITY_CHARS.len() - 1) as f32) as usize;
        let fill = DENSITY_CHARS[idx];
        for row in grid.iter_mut() {
            for cell in row.iter_mut() {
                if *cell == ' ' {
                    *cell = fill;
                }
            }
        }
    }

    fn _render_sprite(grid: &mut Vec<Vec<char>>, layer: &CanvasLayer, width: u16, height: u16) {
        if layer.asset.is_empty() { return; }
        let lines: Vec<&str> = layer.asset.split('\n').collect();
        let start_row = layer.row.max(0) as usize;
        let start_col = layer.col.max(0) as usize;

        for (r, line) in lines.iter().enumerate() {
            let abs_row = start_row + r;
            if abs_row >= height as usize { break; }
            for (c, ch) in line.chars().enumerate() {
                let abs_col = start_col + c;
                if abs_col >= width as usize { break; }
                if ch != ' ' {
                    grid[abs_row][abs_col] = ch;
                }
            }
        }
    }
}
