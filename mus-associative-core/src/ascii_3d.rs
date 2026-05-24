use std::f32::consts::PI;

// ─── ASCII shade ramp ────────────────────────────────────────
pub const SHADE_RAMP: &[u8] = b" .:-=+*#%@";
pub const SHADE_STEPS: usize = 10;

fn shade(value: f32) -> char {
    let idx = ((value.clamp(0.0, 1.0) * (SHADE_STEPS - 1) as f32).round() as usize).min(SHADE_STEPS - 1);
    SHADE_RAMP[idx] as char
}

// ─── 3D math ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Vec3(pub f32, pub f32, pub f32);

impl Vec3 {
    pub fn dot(self, other: Vec3) -> f32 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }
    pub fn sub(self, other: Vec3) -> Vec3 { Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2) }
    pub fn add(self, other: Vec3) -> Vec3 { Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2) }
    pub fn mul(self, s: f32) -> Vec3 { Vec3(self.0 * s, self.1 * s, self.2 * s) }
    pub fn len(self) -> f32 { (self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt() }
    pub fn norm(self) -> Vec3 {
        let l = self.len();
        if l == 0.0 { Vec3(0.0, 0.0, 1.0) } else { self.mul(1.0 / l) }
    }
    pub fn neg(self) -> Vec3 { Vec3(-self.0, -self.1, -self.2) }
}

#[derive(Debug, Clone, Copy)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    pub fn identity() -> Mat4 {
        Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = target.sub(eye).norm();
        let s = f.cross(up).norm();
        let u = s.cross(f);
        Mat4([
            s.0, u.0, -f.0, 0.0,
            s.1, u.1, -f.1, 0.0,
            s.2, u.2, -f.2, 0.0,
            -s.dot(eye), -u.dot(eye), f.dot(eye), 1.0,
        ])
    }
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov * 0.5).tan();
        let nf = 1.0 / (near - far);
        Mat4([
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) * nf, -1.0,
            0.0, 0.0, 2.0 * far * near * nf, 0.0,
        ])
    }
    pub fn rotation_y(angle: f32) -> Mat4 {
        let c = angle.cos();
        let s = angle.sin();
        Mat4([
            c, 0.0, s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }
    pub fn rotation_x(angle: f32) -> Mat4 {
        let c = angle.cos();
        let s = angle.sin();
        Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, c, -s, 0.0,
            0.0, s, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }
    pub fn transform(&self, v: Vec3) -> Vec3 {
        let x = self.0[0] * v.0 + self.0[1] * v.1 + self.0[2] * v.2 + self.0[3];
        let y = self.0[4] * v.0 + self.0[5] * v.1 + self.0[6] * v.2 + self.0[7];
        let z = self.0[8] * v.0 + self.0[9] * v.1 + self.0[10] * v.2 + self.0[11];
        let w = self.0[12] * v.0 + self.0[13] * v.1 + self.0[14] * v.2 + self.0[15];
        if w != 0.0 { Vec3(x / w, y / w, z / w) } else { Vec3(x, y, z) }
    }
}

// ─── Camera ──────────────────────────────────────────────────

pub struct Camera {
    pub pos: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(pos: Vec3, target: Vec3) -> Self {
        Camera { pos, target, up: Vec3(0.0, 1.0, 0.0), fov: PI / 3.0, near: 0.1, far: 100.0 }
    }
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at(self.pos, self.target, self.up);
        let proj = Mat4::perspective(self.fov, aspect, self.near, self.far);
        // combine: proj * view
        Mat4::identity() // simplified — proj * view in shader
    }
}

// ─── Z-buffer / Framebuffer ──────────────────────────────────

pub struct ZBuffer {
    pub width: usize,
    pub height: usize,
    chars: Vec<u8>,
    depth: Vec<f32>,
}

impl ZBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        ZBuffer {
            width, height,
            chars: vec![b' '; width * height],
            depth: vec![f32::INFINITY; width * height],
        }
    }

    pub fn clear(&mut self) {
        self.chars.fill(b' ');
        self.depth.fill(f32::INFINITY);
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, z: f32, ch: u8) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 { return; }
        let idx = y as usize * self.width + x as usize;
        if z < self.depth[idx] {
            self.depth[idx] = z;
            self.chars[idx] = ch;
        }
    }

    pub fn project(&self, v: Vec3, view_proj: &Mat4, screen_scale: f32) -> Option<(i32, i32, f32)> {
        let tv = view_proj.transform(v);
        if tv.2.abs() < 0.001 { return None; }
        let x = ((tv.0 / tv.2) * screen_scale + self.width as f32 * 0.5) as i32;
        let y = ((-tv.1 / tv.2) * screen_scale + self.height as f32 * 0.5) as i32;
        let z = tv.2;
        Some((x, y, z))
    }

    pub fn draw_line(&mut self, a: (i32, i32, f32), b: (i32, i32, f32), ch: u8) {
        let (x0, y0, z0) = a;
        let (x1, y1, z1) = b;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = x0;
        let mut y = y0;
        let steps = dx.max(dy.abs()) as f32;
        loop {
            let t = if steps > 0.0 {
                let d = (dx + dy.abs()) as f32;
                if d > 0.0 { (dx as f32 - (x - x0).abs() as f32) / d } else { 0.5 }
            } else { 0.5 };
            let z = z0 + (z1 - z0) * t;
            self.set_pixel(x, y, z, ch);
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.width * self.height + self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                out.push(self.chars[y * self.width + x] as char);
            }
            if y + 1 < self.height { out.push('\n'); }
        }
        out
    }

    pub fn shade_pixel(&mut self, x: i32, y: i32, z: f32, brightness: f32) {
        let ch = shade(brightness);
        self.set_pixel(x, y, z, ch as u8);
    }
}

// ─── 3D Objects ──────────────────────────────────────────────

pub fn cube_vertices(size: f32) -> Vec<(Vec3, Vec3)> {
    let s = size * 0.5;
    let v = [
        Vec3(-s, -s, -s), Vec3( s, -s, -s), Vec3( s,  s, -s), Vec3(-s,  s, -s),
        Vec3(-s, -s,  s), Vec3( s, -s,  s), Vec3( s,  s,  s), Vec3(-s,  s,  s),
    ];
    let edges = [
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ];
    edges.iter().map(|&(a,b)| (v[a], v[b])).collect()
}

pub fn sphere_vertices(radius: f32, rings: usize, sectors: usize) -> Vec<(Vec3, Vec3)> {
    let mut edges = Vec::new();
    let mut points = Vec::new();
    for r in 0..=rings {
        let theta = r as f32 * PI / rings as f32;
        for s in 0..sectors {
            let phi = s as f32 * 2.0 * PI / sectors as f32;
            let x = theta.sin() * phi.cos() * radius;
            let y = theta.cos() * radius;
            let z = theta.sin() * phi.sin() * radius;
            points.push(Vec3(x, y, z));
        }
    }
    for r in 0..rings {
        for s in 0..sectors {
            let cur = r * sectors + s;
            let next = cur + sectors;
            let s_next = if s + 1 < sectors { s + 1 } else { 0 };
            edges.push((points[cur], points[r * sectors + s_next]));
            edges.push((points[cur], points[next]));
            if s_next > s {
                edges.push((points[next], points[next + 1]));
            }
        }
    }
    edges
}

pub fn torus_vertices(major: f32, minor: f32, rings: usize, sectors: usize) -> Vec<(Vec3, Vec3)> {
    let mut edges = Vec::new();
    let mut points = Vec::new();
    for r in 0..rings {
        let theta = r as f32 * 2.0 * PI / rings as f32;
        for s in 0..sectors {
            let phi = s as f32 * 2.0 * PI / sectors as f32;
            let x = (major + minor * phi.cos()) * theta.cos();
            let y = minor * phi.sin();
            let z = (major + minor * phi.cos()) * theta.sin();
            points.push(Vec3(x, y, z));
        }
    }
    for r in 0..rings {
        for s in 0..sectors {
            let cur = r * sectors + s;
            let next_r = ((r + 1) % rings) * sectors + s;
            let next_s = r * sectors + ((s + 1) % sectors);
            edges.push((points[cur], points[next_r]));
            edges.push((points[cur], points[next_s]));
        }
    }
    edges
}

// ─── Graph → 3D mapping ──────────────────────────────────────

pub struct GraphPoint {
    pub pos: Vec3,
    pub energy: f32,
    pub label: String,
}

/// Map graph node indices to Fibonacci sphere points with energy shading
pub fn map_graph_to_3d(concepts: &[(u32, f32)], total_count: usize, scale: f32) -> Vec<GraphPoint> {
    let golden_angle = PI * (3.0 - (5.0f32).sqrt());
    let mut points = Vec::with_capacity(concepts.len());

    for (i, &(_node_id, energy)) in concepts.iter().enumerate() {
        let phi = i as f32 * golden_angle;
        let z = 1.0 - (i as f32 / (total_count.max(1) - 1) as f32) * 2.0;
        let radius = (1.0 - z * z).sqrt() * scale;
        let x = radius * phi.cos();
        let y = radius * phi.sin();
        let z_coord = z * scale * 0.7;
        points.push(GraphPoint {
            pos: Vec3(x, y, z_coord),
            energy,
            label: String::new(),
        });
    }
    points
}

pub fn render_graph_scene(
    width: usize,
    height: usize,
    points: &[GraphPoint],
    edges: &[(usize, usize)],
    rot_angle: f32,
    seed_idx: Option<usize>,
) -> String {
    let mut zbuf = ZBuffer::new(width, height);
    let aspect = width as f32 / height as f32;
    let cam_dist = 50.0;
    let camera = Camera::new(
        Vec3(cam_dist * rot_angle.sin(), cam_dist * 0.3, cam_dist * rot_angle.cos()),
        Vec3(0.0, 0.0, 0.0),
    );
    let view = Mat4::look_at(camera.pos, camera.target, camera.up);
    let proj = Mat4::perspective(camera.fov, aspect, camera.near, camera.far);

    // Composite view-projection
    // For each point: p_view = view * p, then project with proj
    let transformed: Vec<(f32, f32, f32, f32)> = points.iter().map(|gp| {
        let pv = view.transform(gp.pos);
        (pv.0, pv.1, pv.2, gp.energy)
    }).collect();

    // Draw edges (wireframe)
    for &(ai, bi) in edges {
        if ai >= transformed.len() || bi >= transformed.len() { continue; }
        let (ax, ay, az, _) = transformed[ai];
        let (bx, by, bz, _) = transformed[bi];
        let pa = proj.transform(Vec3(ax, ay, az));
        let pb = proj.transform(Vec3(bx, by, bz));
        if pa.2.abs() < 0.001 || pb.2.abs() < 0.001 { continue; }
        let sx_a = (pa.0 / pa.2 * 10.0 + width as f32 * 0.5) as i32;
        let sy_a = (-pa.1 / pa.2 * 10.0 + height as f32 * 0.5) as i32;
        let sx_b = (pb.0 / pb.2 * 10.0 + width as f32 * 0.5) as i32;
        let sy_b = (-pb.1 / pb.2 * 10.0 + height as f32 * 0.5) as i32;
        let za = pa.2;
        let zb = pb.2;
        zbuf.draw_line((sx_a, sy_a, za), (sx_b, sy_b, zb), b'.');
    }

    // Draw nodes with energy shading
    for (i, &(wx, wy, wz, energy)) in transformed.iter().enumerate() {
        let pp = proj.transform(Vec3(wx, wy, wz));
        if pp.2.abs() < 0.001 { continue; }
        let sx = (pp.0 / pp.2 * 10.0 + width as f32 * 0.5) as i32;
        let sy = (-pp.1 / pp.2 * 10.0 + height as f32 * 0.5) as i32;
        let ch = shade(energy);
        let is_seed = seed_idx.map(|si| si == i).unwrap_or(false);
        zbuf.set_pixel(sx, sy, pp.2, if is_seed { b'@' } else { ch as u8 });
        // Draw a halo of 4 extra dim pixels for visibility
        zbuf.set_pixel(sx - 1, sy, pp.2, shade(energy * 0.3) as u8);
        zbuf.set_pixel(sx + 1, sy, pp.2, shade(energy * 0.3) as u8);
        zbuf.set_pixel(sx, sy - 1, pp.2, shade(energy * 0.3) as u8);
        zbuf.set_pixel(sx, sy + 1, pp.2, shade(energy * 0.3) as u8);
    }

    zbuf.render()
}

// ─── Legacy helpers ──────────────────────────────────────────

pub fn render_edges(
    zbuf: &mut ZBuffer,
    edges: &[(Vec3, Vec3)],
    view_proj: &Mat4,
    scale: f32,
    ch: u8,
) {
    for (a, b) in edges {
        if let Some(pa) = zbuf.project(*a, view_proj, scale) {
            if let Some(pb) = zbuf.project(*b, view_proj, scale) {
                zbuf.draw_line(pa, pb, ch);
            }
        }
    }
}

pub fn render_wireframe_cube(
    zbuf: &mut ZBuffer,
    camera: &Camera,
    size: f32,
    rot_angle: f32,
    aspect: f32,
) {
    let edges = cube_vertices(size);
    let rot_y = Mat4::rotation_y(rot_angle);
    let rot_x = Mat4::rotation_x(rot_angle * 0.3);
    let view = Mat4::look_at(camera.pos, camera.target, camera.up);
    let proj = Mat4::perspective(camera.fov, aspect, camera.near, camera.far);

    // transform edges by rotation
    let transformed: Vec<(Vec3, Vec3)> = edges.iter().map(|(a, b)| {
        let ra = rot_y.transform(rot_x.transform(*a));
        let rb = rot_y.transform(rot_x.transform(*b));
        (ra, rb)
    }).collect();

    for (a, b) in &transformed {
        let wa = view.transform(*a);
        let wb = view.transform(*b);
        if let Some(pa) = zbuf.project(wa, &proj, 10.0) {
            if let Some(pb) = zbuf.project(wb, &proj, 10.0) {
                zbuf.draw_line(pa, pb, b'#');
            }
        }
    }
}

pub fn render_scene_3d(
    width: usize,
    height: usize,
    rot_angle: f32,
) -> String {
    let mut zbuf = ZBuffer::new(width, height);
    let camera = Camera::new(Vec3(3.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0));
    let aspect = width as f32 / height as f32;

    let edges = cube_vertices(2.0);
    let rot_y = Mat4::rotation_y(rot_angle);
    let rot_x = Mat4::rotation_x(rot_angle * 0.5);
    let view = Mat4::look_at(camera.pos, camera.target, camera.up);
    let proj = Mat4::perspective(camera.fov, aspect, camera.near, camera.far);

    let transformed: Vec<(Vec3, Vec3)> = edges.iter().map(|(a, b)| {
        let ra = rot_y.transform(rot_x.transform(*a));
        let rb = rot_y.transform(rot_x.transform(*b));
        (ra, rb)
    }).collect();

    for (a, b) in &transformed {
        let wa = view.transform(*a);
        let wb = view.transform(*b);
        let proj_view = Mat4::identity();
        if let Some(pa) = zbuf.project(wa, &proj_view, 10.0) {
            if let Some(pb) = zbuf.project(wb, &proj_view, 10.0) {
                zbuf.draw_line(pa, pb, b'#');
            }
        }
    }

    zbuf.render()
}
