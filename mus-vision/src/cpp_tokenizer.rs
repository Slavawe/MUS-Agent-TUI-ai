use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Uragan 1.0 C++ Vision Tokenizer — высокопроизводительный токенизатор.
///
/// Режимы:
/// - `Photo` — perceptual pipeline: gamma sRGB → BT.709 luminance → sqrt quantization
/// - `Graph` — edge pipeline: raw luminance → Sobel edge detection → linear quantization

const DEFAULT_PALETTE: &str = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#8&@%░▒▓█▄▀■¶§°•";

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum EncodeMode {
    Photo = 0,
    Graph = 1,
}

#[pymethods]
impl EncodeMode {
    #[new]
    fn py_new(mode: &str) -> PyResult<Self> {
        match mode.to_lowercase().as_str() {
            "photo" => Ok(EncodeMode::Photo),
            "graph" => Ok(EncodeMode::Graph),
            _ => Err(PyValueError::new_err(format!(
                "Unknown EncodeMode '{}', expected 'photo' or 'graph'", mode
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            EncodeMode::Photo => "EncodeMode.Photo".to_string(),
            EncodeMode::Graph => "EncodeMode.Graph".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            EncodeMode::Photo => "photo".to_string(),
            EncodeMode::Graph => "graph".to_string(),
        }
    }
}

#[pyclass]
pub struct CPPTokenizer {
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
    #[pyo3(get)]
    pub palette: String,
    #[pyo3(get)]
    pub cpp_start: i64,
    #[pyo3(get)]
    pub vision_start: i64,
    #[pyo3(get)]
    pub vision_end: i64,
    #[pyo3(get)]
    pub frame_sep: i64,
    #[pyo3(get)]
    pub mode: EncodeMode,
    palette_chars: Vec<char>,
    palette_len: usize,
    gamma_lut: [f32; 256],
    quant_lut: [usize; 256],
}

#[pymethods]
impl CPPTokenizer {
    #[new]
    #[pyo3(signature = (width=64, height=32, palette=None, cpp_start=2001, vision_start=2101, vision_end=2102, frame_sep=2103, mode="photo"))]
    pub fn new(
        width: usize,
        height: usize,
        palette: Option<String>,
        cpp_start: i64,
        vision_start: i64,
        vision_end: i64,
        frame_sep: i64,
        mode: &str,
    ) -> PyResult<Self> {
        let enc_mode = EncodeMode::py_new(mode)?;
        let palette = palette.unwrap_or_else(|| DEFAULT_PALETTE.to_string());
        let palette_len = palette.chars().count();
        let palette_chars: Vec<char> = palette.chars().collect();

        let mut gamma_lut = [0.0f32; 256];
        for i in 0..256 {
            let v = i as f32 / 255.0;
            gamma_lut[i] = if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            };
        }

        let mut quant_lut = [0usize; 256];
        for i in 0..256 {
            let linear = gamma_lut[i];
            let perceptual = linear.sqrt();
            quant_lut[i] = (perceptual * (palette_len - 1) as f32)
                .round()
                .max(0.0)
                .min((palette_len - 1) as f32) as usize;
        }

        Ok(Self {
            width,
            height,
            palette,
            palette_chars,
            palette_len,
            cpp_start,
            vision_start,
            vision_end,
            frame_sep,
            mode: enc_mode,
            gamma_lut,
            quant_lut,
        })
    }

    pub fn set_mode(&mut self, mode: &str) -> PyResult<()> {
        self.mode = EncodeMode::py_new(mode)?;
        Ok(())
    }

    pub fn encode_image(&self, pixels: Vec<u8>, width_in: usize, height_in: usize) -> PyResult<Vec<i64>> {
        if pixels.len() != width_in * height_in * 3 && pixels.len() != width_in * height_in {
            return Err(PyValueError::new_err("Pixel data size mismatch"));
        }

        let is_rgb = pixels.len() == width_in * height_in * 3;

        match self.mode {
            EncodeMode::Photo => {
                if is_rgb {
                    self.encode_from_slice_rgb(&pixels, width_in, height_in)
                } else {
                    self.encode_from_slice_gray(&pixels, width_in, height_in)
                }
            }
            EncodeMode::Graph => {
                if is_rgb {
                    self.encode_graph_rgb(&pixels, width_in, height_in)
                } else {
                    self.encode_graph_gray(&pixels, width_in, height_in)
                }
            }
        }
    }

    pub fn encode_video(
        &self,
        frames: Vec<Vec<u8>>,
        frame_widths: Vec<usize>,
        frame_heights: Vec<usize>,
    ) -> PyResult<Vec<i64>> {
        let n = frames.len();
        let mut tokens = Vec::new();
        tokens.push(self.vision_start);

        for i in 0..n {
            let frame_tokens = self.encode_image(frames[i].clone(), frame_widths[i], frame_heights[i])?;
            tokens.extend_from_slice(&frame_tokens[1..frame_tokens.len() - 1]);
            if i < n - 1 {
                tokens.push(self.frame_sep);
            }
        }
        tokens.push(self.vision_end);
        Ok(tokens)
    }

    #[pyo3(signature = (frames, frame_widths, frame_heights, threshold=0.1))]
    pub fn encode_video_diff(
        &self,
        frames: Vec<Vec<u8>>,
        frame_widths: Vec<usize>,
        frame_heights: Vec<usize>,
        threshold: f32,
    ) -> PyResult<Vec<i64>> {
        let n = frames.len();
        let mut tokens = Vec::new();
        tokens.push(self.vision_start);

        let mut prev_frame: Option<Vec<f32>> = None;

        for i in 0..n {
            let resized = self.resize_to_f32(&frames[i], frame_widths[i], frame_heights[i], &self.mode)?;

            if i == 0 || prev_frame.is_none() {
                for &val in &resized {
                    let qi = quantize(val, self.palette_len, &self.mode);
                    tokens.push(self.cpp_start + qi as i64);
                }
                prev_frame = Some(resized);
            } else {
                let prev = prev_frame.as_ref().unwrap();
                let mut changed_count = 0u32;
                let mut changed_data = Vec::new();

                for (idx, (&cur, &prev_val)) in resized.iter().zip(prev.iter()).enumerate() {
                    if (cur - prev_val).abs() > threshold {
                        changed_count += 1;
                        let qi = quantize(cur, self.palette_len, &self.mode);
                        let y = (idx / self.width) as u16;
                        let x = (idx % self.width) as u16;
                        changed_data.push(y);
                        changed_data.push(x);
                        changed_data.push((self.cpp_start + qi as i64) as u16);
                    }
                }

                tokens.push(changed_count as i64);
                for &v in &changed_data {
                    tokens.push(v as i64);
                }
                prev_frame = Some(resized);
            }

            if i < n - 1 {
                tokens.push(self.frame_sep);
            }
        }
        tokens.push(self.vision_end);
        Ok(tokens)
    }

    pub fn decode(&self, tokens: Vec<i64>) -> String {
        let start_idx = if !tokens.is_empty() && tokens[0] == self.vision_start { 1 } else { 0 };
        let mut result = String::with_capacity(self.height * (self.width + 4));

        for y in 0..self.height {
            if y > 0 {
                result.push('\n');
            }
            for x in 0..self.width {
                let token_idx = start_idx + y * self.width + x;
                let token = if token_idx < tokens.len() { tokens[token_idx] } else { self.cpp_start };
                let pi = ((token - self.cpp_start).max(0) as usize).min(self.palette_len - 1);
                result.push(self.palette_chars[pi]);
            }
        }
        result
    }

    pub fn decode_video(&self, tokens: Vec<i64>) -> Vec<String> {
        let mut frames = Vec::new();
        let mut current = Vec::new();

        for &t in &tokens {
            if t == self.frame_sep {
                if !current.is_empty() {
                    frames.push(self.decode(current.clone()));
                    current.clear();
                }
            } else if t != self.vision_start && t != self.vision_end {
                current.push(t);
            }
        }
        if !current.is_empty() {
            frames.push(self.decode(current));
        }
        frames
    }

    pub fn generate_shape(&self, shape: &str, position: &str, size: &str) -> Vec<i64> {
        let cx = self.position_x(position);
        let cy = self.position_y(position);
        let r = self.size_radius(size);

        let mut tokens = Vec::with_capacity(self.width * self.height + 2);
        tokens.push(self.vision_start);

        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let val = match shape {
                    "circle" => (1.0 - (dx * dx + dy * dy).sqrt() / r).max(0.0),
                    "square" => (1.0 - dx.abs().max(dy.abs()) / r).max(0.0),
                    "diamond" => (1.0 - (dx.abs() / r + dy.abs() / r - 0.3).max(0.0)).max(0.0),
                    "gradient" => (x as f32 / self.width as f32 + y as f32 / self.height as f32) / 2.0,
                    "waves" => 0.5 + 0.5 * (x as f32 * 0.5 + y as f32 * 0.3).sin(),
                    "checkerboard" => if ((x + y) % 4) < 2 { 1.0 } else { 0.0 },
                    "cross" => if dx.abs() < 2.0 || dy.abs() < 2.0 {
                        (1.0 - dx.abs().min(dy.abs()) / 3.0).max(0.0)
                    } else { 0.0 },
                    "triangle" => {
                        let prog = y as f32 / self.height as f32;
                        let hw = r * (1.0 - prog);
                        if dx.abs() < hw { (1.0 - dx.abs() / hw.max(0.01)).max(0.0) } else { 0.0 }
                    }
                    "spiral" => {
                        let d = (dx * dx + dy * dy).sqrt();
                        let angle = dy.atan2(dx);
                        (0.5 + 0.5 * (d * 0.8 - angle * 3.0).sin()).max(0.0).min(1.0)
                    }
                    _ => 0.0,
                };
                let qi = quantize(val.clamp(0.0, 1.0), self.palette_len, &self.mode);
                tokens.push(self.cpp_start + qi as i64);
            }
        }
        tokens.push(self.vision_end);
        tokens
    }

    pub fn info(&self) -> String {
        format!(
            "CPPTokenizer v1.0 (Uragan): {}x{}, palette={} chars, mode={}, tokens {}-{}",
            self.width, self.height, self.palette_len, self.mode.__str__(),
            self.cpp_start, self.cpp_start + self.palette_len as i64 - 1
        )
    }
}

impl CPPTokenizer {
    fn encode_from_slice_rgb(&self, pixels: &[u8], w_in: usize, h_in: usize) -> PyResult<Vec<i64>> {
        let total = self.width * self.height;
        let mut tokens = Vec::with_capacity(total + 2);
        tokens.push(self.vision_start);

        let rx = w_in as f32 / self.width as f32;
        let ry = h_in as f32 / self.height as f32;

        let row_offsets: Vec<usize> = (0..h_in).map(|y| y * w_in * 3).collect();

        for y in 0..self.height {
            let sy_f = y as f32 * ry;
            let sy = (sy_f as usize).min(h_in - 2);
            let fy = sy_f - sy as f32;
            let fy_inv = 1.0 - fy;

            for x in 0..self.width {
                let sx_f = x as f32 * rx;
                let sx = (sx_f as usize).min(w_in - 2);
                let fx = sx_f - sx as f32;
                let fx_inv = 1.0 - fx;

                let base00 = row_offsets[sy] + sx * 3;
                let base10 = row_offsets[sy + 1] + sx * 3;
                let base01 = base00 + 3;
                let base11 = base10 + 3;

                let g00 = self.gamma_lut[pixels[base00] as usize] * 0.2126
                    + self.gamma_lut[pixels[base00 + 1] as usize] * 0.7152
                    + self.gamma_lut[pixels[base00 + 2] as usize] * 0.0722;
                let g10 = self.gamma_lut[pixels[base10] as usize] * 0.2126
                    + self.gamma_lut[pixels[base10 + 1] as usize] * 0.7152
                    + self.gamma_lut[pixels[base10 + 2] as usize] * 0.0722;
                let g01 = self.gamma_lut[pixels[base01] as usize] * 0.2126
                    + self.gamma_lut[pixels[base01 + 1] as usize] * 0.7152
                    + self.gamma_lut[pixels[base01 + 2] as usize] * 0.0722;
                let g11 = self.gamma_lut[pixels[base11] as usize] * 0.2126
                    + self.gamma_lut[pixels[base11 + 1] as usize] * 0.7152
                    + self.gamma_lut[pixels[base11 + 2] as usize] * 0.0722;

                let linear = g00 * fy_inv * fx_inv
                    + g10 * fy * fx_inv
                    + g01 * fy_inv * fx
                    + g11 * fy * fx;

                let perceptual = linear.sqrt();
                let qi = (perceptual * (self.palette_len - 1) as f32)
                    .round()
                    .max(0.0)
                    .min((self.palette_len - 1) as f32) as usize;

                tokens.push(self.cpp_start + qi as i64);
            }
        }

        tokens.push(self.vision_end);
        Ok(tokens)
    }

    fn encode_from_slice_gray(&self, pixels: &[u8], w_in: usize, h_in: usize) -> PyResult<Vec<i64>> {
        let total = self.width * self.height;
        let mut tokens = Vec::with_capacity(total + 2);
        tokens.push(self.vision_start);

        let rx = w_in as f32 / self.width as f32;
        let ry = h_in as f32 / self.height as f32;

        let row_offsets: Vec<usize> = (0..h_in).map(|y| y * w_in).collect();

        for y in 0..self.height {
            let sy_f = y as f32 * ry;
            let sy = (sy_f as usize).min(h_in - 2);
            let fy = sy_f - sy as f32;
            let fy_inv = 1.0 - fy;

            for x in 0..self.width {
                let sx_f = x as f32 * rx;
                let sx = (sx_f as usize).min(w_in - 2);
                let fx = sx_f - sx as f32;
                let fx_inv = 1.0 - fx;

                let base00 = row_offsets[sy] + sx;
                let base10 = row_offsets[sy + 1] + sx;
                let base01 = base00 + 1;
                let base11 = base10 + 1;

                let g00 = self.gamma_lut[pixels[base00] as usize];
                let g10 = self.gamma_lut[pixels[base10] as usize];
                let g01 = self.gamma_lut[pixels[base01] as usize];
                let g11 = self.gamma_lut[pixels[base11] as usize];

                let linear = g00 * fy_inv * fx_inv
                    + g10 * fy * fx_inv
                    + g01 * fy_inv * fx
                    + g11 * fy * fx;

                let perceptual = linear.sqrt();
                let qi = (perceptual * (self.palette_len - 1) as f32)
                    .round()
                    .max(0.0)
                    .min((self.palette_len - 1) as f32) as usize;

                tokens.push(self.cpp_start + qi as i64);
            }
        }

        tokens.push(self.vision_end);
        Ok(tokens)
    }

    fn encode_graph_rgb(&self, pixels: &[u8], w_in: usize, h_in: usize) -> PyResult<Vec<i64>> {
        let total = self.width * self.height;
        let mut luminance = vec![0.0f32; total];

        let rx = w_in as f32 / self.width as f32;
        let ry = h_in as f32 / self.height as f32;

        let row_offsets: Vec<usize> = (0..h_in).map(|y| y * w_in * 3).collect();

        for y in 0..self.height {
            let sy_f = y as f32 * ry;
            let sy = (sy_f as usize).min(h_in - 2);
            let fy = sy_f - sy as f32;
            let fy_inv = 1.0 - fy;

            for x in 0..self.width {
                let sx_f = x as f32 * rx;
                let sx = (sx_f as usize).min(w_in - 2);
                let fx = sx_f - sx as f32;
                let fx_inv = 1.0 - fx;

                let base00 = row_offsets[sy] + sx * 3;
                let base10 = row_offsets[sy + 1] + sx * 3;
                let base01 = base00 + 3;
                let base11 = base10 + 3;

                let l00 = pixels[base00] as f32 / 255.0 * 0.2126
                    + pixels[base00 + 1] as f32 / 255.0 * 0.7152
                    + pixels[base00 + 2] as f32 / 255.0 * 0.0722;
                let l10 = pixels[base10] as f32 / 255.0 * 0.2126
                    + pixels[base10 + 1] as f32 / 255.0 * 0.7152
                    + pixels[base10 + 2] as f32 / 255.0 * 0.0722;
                let l01 = pixels[base01] as f32 / 255.0 * 0.2126
                    + pixels[base01 + 1] as f32 / 255.0 * 0.7152
                    + pixels[base01 + 2] as f32 / 255.0 * 0.0722;
                let l11 = pixels[base11] as f32 / 255.0 * 0.2126
                    + pixels[base11 + 1] as f32 / 255.0 * 0.7152
                    + pixels[base11 + 2] as f32 / 255.0 * 0.0722;

                let raw = l00 * fy_inv * fx_inv
                    + l10 * fy * fx_inv
                    + l01 * fy_inv * fx
                    + l11 * fy * fx;

                luminance[y * self.width + x] = raw;
            }
        }

        let edges = sobel_edges(&luminance, self.width, self.height);

        let mut tokens = Vec::with_capacity(total + 2);
        tokens.push(self.vision_start);
        for &mag in &edges {
            let qi = quantize_linear(mag, self.palette_len);
            tokens.push(self.cpp_start + qi as i64);
        }
        tokens.push(self.vision_end);
        Ok(tokens)
    }

    fn encode_graph_gray(&self, pixels: &[u8], w_in: usize, h_in: usize) -> PyResult<Vec<i64>> {
        let total = self.width * self.height;
        let mut luminance = vec![0.0f32; total];

        let rx = w_in as f32 / self.width as f32;
        let ry = h_in as f32 / self.height as f32;

        let row_offsets: Vec<usize> = (0..h_in).map(|y| y * w_in).collect();

        for y in 0..self.height {
            let sy_f = y as f32 * ry;
            let sy = (sy_f as usize).min(h_in - 2);
            let fy = sy_f - sy as f32;
            let fy_inv = 1.0 - fy;

            for x in 0..self.width {
                let sx_f = x as f32 * rx;
                let sx = (sx_f as usize).min(w_in - 2);
                let fx = sx_f - sx as f32;
                let fx_inv = 1.0 - fx;

                let base00 = row_offsets[sy] + sx;
                let base10 = row_offsets[sy + 1] + sx;
                let base01 = base00 + 1;
                let base11 = base10 + 1;

                let l00 = pixels[base00] as f32 / 255.0;
                let l10 = pixels[base10] as f32 / 255.0;
                let l01 = pixels[base01] as f32 / 255.0;
                let l11 = pixels[base11] as f32 / 255.0;

                let raw = l00 * fy_inv * fx_inv
                    + l10 * fy * fx_inv
                    + l01 * fy_inv * fx
                    + l11 * fy * fx;

                luminance[y * self.width + x] = raw;
            }
        }

        let edges = sobel_edges(&luminance, self.width, self.height);

        let mut tokens = Vec::with_capacity(total + 2);
        tokens.push(self.vision_start);
        for &mag in &edges {
            let qi = quantize_linear(mag, self.palette_len);
            tokens.push(self.cpp_start + qi as i64);
        }
        tokens.push(self.vision_end);
        Ok(tokens)
    }

    fn resize_to_f32(&self, pixels: &[u8], w_in: usize, h_in: usize, mode: &EncodeMode) -> PyResult<Vec<f32>> {
        let is_rgb = pixels.len() == w_in * h_in * 3;
        let use_gamma = matches!(mode, EncodeMode::Photo);
        let mut resized = Vec::with_capacity(self.width * self.height);
        let rx = w_in as f32 / self.width as f32;
        let ry = h_in as f32 / self.height as f32;

        let row_offsets: Vec<usize> = if is_rgb {
            (0..h_in).map(|y| y * w_in * 3).collect()
        } else {
            (0..h_in).map(|y| y * w_in).collect()
        };

        for y in 0..self.height {
            let sy_f = y as f32 * ry;
            let sy = (sy_f as usize).min(h_in - 2);
            let fy = sy_f - sy as f32;
            let fy_inv = 1.0 - fy;

            for x in 0..self.width {
                let sx_f = x as f32 * rx;
                let sx = (sx_f as usize).min(w_in - 2);
                let fx = sx_f - sx as f32;
                let fx_inv = 1.0 - fx;

                let val = if is_rgb {
                    let base00 = row_offsets[sy] + sx * 3;
                    let base10 = row_offsets[sy + 1] + sx * 3;
                    let base01 = base00 + 3;
                    let base11 = base10 + 3;

                    if use_gamma {
                        let g00 = self.gamma_lut[pixels[base00] as usize] * 0.2126
                            + self.gamma_lut[pixels[base00 + 1] as usize] * 0.7152
                            + self.gamma_lut[pixels[base00 + 2] as usize] * 0.0722;
                        let g10 = self.gamma_lut[pixels[base10] as usize] * 0.2126
                            + self.gamma_lut[pixels[base10 + 1] as usize] * 0.7152
                            + self.gamma_lut[pixels[base10 + 2] as usize] * 0.0722;
                        let g01 = self.gamma_lut[pixels[base01] as usize] * 0.2126
                            + self.gamma_lut[pixels[base01 + 1] as usize] * 0.7152
                            + self.gamma_lut[pixels[base01 + 2] as usize] * 0.0722;
                        let g11 = self.gamma_lut[pixels[base11] as usize] * 0.2126
                            + self.gamma_lut[pixels[base11 + 1] as usize] * 0.7152
                            + self.gamma_lut[pixels[base11 + 2] as usize] * 0.0722;

                        g00 * fy_inv * fx_inv + g10 * fy * fx_inv + g01 * fy_inv * fx + g11 * fy * fx
                    } else {
                        let l00 = pixels[base00] as f32 / 255.0 * 0.2126
                            + pixels[base00 + 1] as f32 / 255.0 * 0.7152
                            + pixels[base00 + 2] as f32 / 255.0 * 0.0722;
                        let l10 = pixels[base10] as f32 / 255.0 * 0.2126
                            + pixels[base10 + 1] as f32 / 255.0 * 0.7152
                            + pixels[base10 + 2] as f32 / 255.0 * 0.0722;
                        let l01 = pixels[base01] as f32 / 255.0 * 0.2126
                            + pixels[base01 + 1] as f32 / 255.0 * 0.7152
                            + pixels[base01 + 2] as f32 / 255.0 * 0.0722;
                        let l11 = pixels[base11] as f32 / 255.0 * 0.2126
                            + pixels[base11 + 1] as f32 / 255.0 * 0.7152
                            + pixels[base11 + 2] as f32 / 255.0 * 0.0722;

                        l00 * fy_inv * fx_inv + l10 * fy * fx_inv + l01 * fy_inv * fx + l11 * fy * fx
                    }
                } else {
                    let base00 = row_offsets[sy] + sx;
                    let base10 = row_offsets[sy + 1] + sx;
                    let base01 = base00 + 1;
                    let base11 = base10 + 1;

                    if use_gamma {
                        let g00 = self.gamma_lut[pixels[base00] as usize];
                        let g10 = self.gamma_lut[pixels[base10] as usize];
                        let g01 = self.gamma_lut[pixels[base01] as usize];
                        let g11 = self.gamma_lut[pixels[base11] as usize];

                        g00 * fy_inv * fx_inv + g10 * fy * fx_inv + g01 * fy_inv * fx + g11 * fy * fx
                    } else {
                        let l00 = pixels[base00] as f32 / 255.0;
                        let l10 = pixels[base10] as f32 / 255.0;
                        let l01 = pixels[base01] as f32 / 255.0;
                        let l11 = pixels[base11] as f32 / 255.0;

                        l00 * fy_inv * fx_inv + l10 * fy * fx_inv + l01 * fy_inv * fx + l11 * fy * fx
                    }
                };
                resized.push(val);
            }
        }
        Ok(resized)
    }

    fn position_x(&self, pos: &str) -> f32 {
        let m = 0.2;
        match pos {
            "center" => self.width as f32 / 2.0,
            "top_left" | "top left" => self.width as f32 * m,
            "top_right" | "top right" => self.width as f32 * (1.0 - m),
            "bottom_left" | "bottom left" => self.width as f32 * m,
            "bottom_right" | "bottom right" => self.width as f32 * (1.0 - m),
            "left" | "left side" => self.width as f32 * m,
            "right" | "right side" => self.width as f32 * (1.0 - m),
            _ => self.width as f32 / 2.0,
        }
    }

    fn position_y(&self, pos: &str) -> f32 {
        let m = 0.2;
        match pos {
            "center" => self.height as f32 / 2.0,
            "top_left" | "top left" => self.height as f32 * m,
            "top_right" | "top right" => self.height as f32 * m,
            "bottom_left" | "bottom left" => self.height as f32 * (1.0 - m),
            "bottom_right" | "bottom right" => self.height as f32 * (1.0 - m),
            "left" | "left side" => self.height as f32 / 2.0,
            "right" | "right side" => self.height as f32 / 2.0,
            "top" => self.height as f32 * m,
            "bottom" => self.height as f32 * (1.0 - m),
            _ => self.height as f32 / 2.0,
        }
    }

    fn size_radius(&self, size: &str) -> f32 {
        let mr = (self.width.min(self.height) as f32) * 0.45;
        match size {
            "tiny" => mr * 0.2,
            "small" => mr * 0.35,
            "medium" => mr * 0.55,
            "large" => mr * 0.75,
            "huge" => mr * 0.95,
            _ => mr * 0.55,
        }
    }
}

#[inline(always)]
fn quantize(val: f32, palette_len: usize, mode: &EncodeMode) -> usize {
    match mode {
        EncodeMode::Photo => {
            let perceptual = val.sqrt();
            (perceptual * (palette_len - 1) as f32)
                .round()
                .max(0.0)
                .min((palette_len - 1) as f32) as usize
        }
        EncodeMode::Graph => quantize_linear(val, palette_len),
    }
}

#[inline(always)]
fn quantize_linear(val: f32, palette_len: usize) -> usize {
    (val * (palette_len - 1) as f32)
        .round()
        .max(0.0)
        .min((palette_len - 1) as f32) as usize
}

/// Sobel edge detection на canvas luminance.
/// Возвращает magnitude, нормализованную в [0, 1].
fn sobel_edges(luminance: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut magnitude = vec![0.0f32; w * h];
    let mut max_mag = 0.0f32;

    for y in 0..h {
        for x in 0..w {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for ky in 0..3 {
                for kx in 0..3 {
                    let py = (y as i32 + ky as i32 - 1).clamp(0, h as i32 - 1) as usize;
                    let px = (x as i32 + kx as i32 - 1).clamp(0, w as i32 - 1) as usize;
                    let val = luminance[py * w + px];

                    let sx = match (kx, ky) {
                        (0, 0) | (0, 2) => -1,
                        (2, 0) | (2, 2) => 1,
                        (0, 1) => -2,
                        (2, 1) => 2,
                        _ => 0,
                    };
                    let sy = match (kx, ky) {
                        (0, 0) | (2, 0) => -1,
                        (0, 2) | (2, 2) => 1,
                        (1, 0) => -2,
                        (1, 2) => 2,
                        _ => 0,
                    };

                    gx += sx as f32 * val;
                    gy += sy as f32 * val;
                }
            }

            let mag = (gx * gx + gy * gy).sqrt();
            magnitude[y * w + x] = mag;
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }

    if max_mag > 1e-6 {
        for v in &mut magnitude {
            *v = (*v / max_mag).min(1.0);
        }
    }

    magnitude
}
