//! Finding the exam sheet in a phone photo and cutting out its cells, in pure
//! Rust so the same code runs on iOS and Android.
//!
//! 1. `detect_page`: the paper is the largest bright, low-saturation region;
//!    lines fitted to its boundary give sub-pixel corners. Shadows and pale
//!    desks make this unreliable, so it only proposes a starting point.
//! 2. `Registrar`: the page outline that makes the photo's dark structure
//!    (tables, frames, text blocks) correlate best with the blank template,
//!    found by coarse-to-fine corner search from the candidate outlines.
//! 3. `Aligner`: per region, the printed table lines of the template are
//!    matched against the photo by row/column darkness profiles, then every
//!    cell edge snaps to the nearest printed line. Darkness uses the maximum
//!    colour channel, so red and blue ink barely register as lines.
//! 4. `warp_quad` cuts each cell straight from the full-resolution photo.

use image::RgbImage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned rectangle in some pixel space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn corners(&self) -> [Point; 4] {
        [
            Point::new(self.x, self.y),
            Point::new(self.x + self.width, self.y),
            Point::new(self.x + self.width, self.y + self.height),
            Point::new(self.x, self.y + self.height),
        ]
    }

    pub fn contains(&self, p: Point) -> bool {
        (self.x..self.x + self.width).contains(&p.x)
            && (self.y..self.y + self.height).contains(&p.y)
    }

    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    fn translated(&self, dx: f64, dy: f64) -> Rect {
        Rect {
            x: self.x + dx,
            y: self.y + dy,
            ..*self
        }
    }
}

/// Projective transform between two planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    m: [[f64; 3]; 3],
}

impl Homography {
    /// The transform taking each `from[i]` to `to[i]`.
    pub fn from_points(from: &[Point; 4], to: &[Point; 4]) -> Option<Self> {
        let mut a = [[0.0_f64; 9]; 8];
        for i in 0..4 {
            let (x, y, u, v) = (from[i].x, from[i].y, to[i].x, to[i].y);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        let h = solve8(a)?;
        Some(Self {
            m: [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        })
    }

    pub fn apply(&self, p: Point) -> Point {
        let m = &self.m;
        let w = m[2][0] * p.x + m[2][1] * p.y + m[2][2];
        Point::new(
            (m[0][0] * p.x + m[0][1] * p.y + m[0][2]) / w,
            (m[1][0] * p.x + m[1][1] * p.y + m[1][2]) / w,
        )
    }
}

fn solve8(mut a: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot_row = a[col];
        for (row, values) in a.iter_mut().enumerate() {
            if row != col {
                let factor = values[col] / pivot_row[col];
                for (value, pivot_value) in values.iter_mut().zip(pivot_row).skip(col) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| a[i][8] / a[i][i]))
}

/// A single-channel float image.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Mean of row `y` between columns `x0..x1` (clamped).
    fn row_mean(&self, y: isize, x0: isize, x1: isize) -> f32 {
        if y < 0 || y >= self.height as isize {
            return 0.0;
        }
        let (x0, x1) = (x0.max(0) as usize, (x1.max(0) as usize).min(self.width));
        if x1 <= x0 {
            return 0.0;
        }
        (x0..x1).map(|x| self.at(x, y as usize)).sum::<f32>() / (x1 - x0) as f32
    }

    fn column_mean(&self, x: isize, y0: isize, y1: isize) -> f32 {
        if x < 0 || x >= self.width as isize {
            return 0.0;
        }
        let (y0, y1) = (y0.max(0) as usize, (y1.max(0) as usize).min(self.height));
        if y1 <= y0 {
            return 0.0;
        }
        (y0..y1).map(|y| self.at(x as usize, y)).sum::<f32>() / (y1 - y0) as f32
    }
}

fn sample_rgb(photo: &RgbImage, x: f64, y: f64) -> [f32; 3] {
    let (w, h) = (photo.width() as f64, photo.height() as f64);
    let x = x.clamp(0.0, w - 1.0);
    let y = y.clamp(0.0, h - 1.0);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = (
        (x0 + 1).min(photo.width() - 1),
        (y0 + 1).min(photo.height() - 1),
    );
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let mut out = [0.0; 3];
    for (c, value) in out.iter_mut().enumerate() {
        let p = |px: u32, py: u32| f32::from(photo.get_pixel(px, py)[c]);
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        *value = top * (1.0 - fy) + bottom * fy;
    }
    out
}

/// Cuts the quadrilateral `quad` (TL, TR, BR, BL in photo pixels) out of the
/// photo as an upright `width` x `height` RGB image.
pub fn warp_quad(photo: &RgbImage, quad: &[Point; 4], width: u32, height: u32) -> RgbImage {
    let target = Rect {
        x: 0.0,
        y: 0.0,
        width: width as f64,
        height: height as f64,
    }
    .corners();
    let Some(map) = Homography::from_points(&target, quad) else {
        return RgbImage::new(width, height);
    };
    RgbImage::from_fn(width, height, |x, y| {
        let source = map.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
        let [r, g, b] = sample_rgb(photo, source.x - 0.5, source.y - 0.5);
        image::Rgb([r.round() as u8, g.round() as u8, b.round() as u8])
    })
}

fn otsu(values: &[f32]) -> f32 {
    let mut histogram = [0_u64; 256];
    for &v in values {
        histogram[v.clamp(0.0, 255.0) as usize] += 1;
    }
    let total = values.len() as f64;
    let weighted: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &n)| i as f64 * n as f64)
        .sum();
    let (mut background, mut background_sum, mut best, mut threshold) = (0.0, 0.0, -1.0, 127.0);
    for (i, &n) in histogram.iter().enumerate() {
        background += n as f64;
        if background == 0.0 || background == total {
            continue;
        }
        background_sum += i as f64 * n as f64;
        let foreground = total - background;
        let mean_b = background_sum / background;
        let mean_f = (weighted - background_sum) / foreground;
        let variance = background * foreground * (mean_b - mean_f).powi(2);
        if variance > best {
            best = variance;
            threshold = i as f64 + 0.5;
        }
    }
    threshold as f32
}

/// Square-window morphology on a binary mask using an integral image.
fn morph(mask: &[bool], width: usize, height: usize, radius: usize, dilate: bool) -> Vec<bool> {
    let stride = width + 1;
    let mut integral = vec![0_u32; stride * (height + 1)];
    for y in 0..height {
        let mut row = 0;
        for x in 0..width {
            row += u32::from(mask[y * width + x]);
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
        }
    }
    let mut out = vec![false; mask.len()];
    for y in 0..height {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(height));
        for x in 0..width {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(width));
            let count = integral[y1 * stride + x1] + integral[y0 * stride + x0]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0];
            let area = ((x1 - x0) * (y1 - y0)) as u32;
            out[y * width + x] = if dilate { count > 0 } else { count == area };
        }
    }
    out
}

fn largest_component(mask: &[bool], width: usize, height: usize) -> Vec<usize> {
    let mut seen = vec![false; mask.len()];
    let mut best = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut component = vec![start];
        seen[start] = true;
        let mut next = 0;
        while next < component.len() {
            let index = component[next];
            next += 1;
            let (x, y) = (index % width, index / width);
            let mut visit = |n: usize| {
                if mask[n] && !seen[n] {
                    seen[n] = true;
                    component.push(n);
                }
            };
            if x > 0 {
                visit(index - 1);
            }
            if x + 1 < width {
                visit(index + 1);
            }
            if y > 0 {
                visit(index - width);
            }
            if y + 1 < height {
                visit(index + width);
            }
        }
        if component.len() > best.len() {
            best = component;
        }
    }
    best
}

fn line_through(points: &[Point]) -> Option<(Point, Point)> {
    if points.len() < 8 {
        return None;
    }
    let n = points.len() as f64;
    let cx = points.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy) = (p.x - cx, p.y - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some((Point::new(cx, cy), Point::new(angle.cos(), angle.sin())))
}

fn intersect(a: (Point, Point), b: (Point, Point)) -> Option<Point> {
    let ((p, r), (q, s)) = (a, b);
    let denominator = r.x * s.y - r.y * s.x;
    if denominator.abs() < 1e-9 {
        return None;
    }
    let t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / denominator;
    Some(Point::new(p.x + t * r.x, p.y + t * r.y))
}

fn distance_to_segment(p: Point, a: Point, b: Point) -> (f64, f64) {
    let (vx, vy) = (b.x - a.x, b.y - a.y);
    let length2 = vx * vx + vy * vy;
    let t = (((p.x - a.x) * vx + (p.y - a.y) * vy) / length2).clamp(0.0, 1.0);
    let (cx, cy) = (a.x + t * vx, a.y + t * vy);
    (((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt(), t)
}

fn quad_area(quad: &[Point; 4]) -> f64 {
    (0..4)
        .map(|i| {
            let (a, b) = (quad[i], quad[(i + 1) % 4]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        .abs()
        / 2.0
}

/// Corners (TL, TR, BR, BL, photo pixels) of the sheet, or `None` when no
/// plausible page outline is found (e.g. white paper on a white table).
pub fn detect_page(photo: &RgbImage) -> Option<[Point; 4]> {
    let (width, height) = (photo.width() as usize, photo.height() as usize);
    let factor = (width.max(height) as f64 / 800.0).ceil().max(1.0) as usize;
    let (w, h) = (width / factor, height / factor);
    if w < 16 || h < 16 {
        return None;
    }
    // Paper-likeness: bright and colourless, averaged over factor x factor blocks.
    let mut paper = vec![0.0_f32; w * h];
    for by in 0..h {
        for bx in 0..w {
            let mut sum = [0.0_f32; 3];
            for y in by * factor..(by + 1) * factor {
                for x in bx * factor..(bx + 1) * factor {
                    let p = photo.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        sum[c] += f32::from(p[c]);
                    }
                }
            }
            let n = (factor * factor) as f32;
            let (r, g, b) = (sum[0] / n, sum[1] / n, sum[2] / n);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };
            paper[by * w + bx] = max * (1.0 - saturation);
        }
    }
    let threshold = otsu(&paper);
    let mask: Vec<bool> = paper.iter().map(|&v| v > threshold).collect();
    let closed = morph(&morph(&mask, w, h, 3, true), w, h, 3, false);
    let component = largest_component(&closed, w, h);
    if component.len() < w * h / 10 {
        return None;
    }

    let in_component = {
        let mut flags = vec![false; w * h];
        for &i in &component {
            flags[i] = true;
        }
        flags
    };
    let point = |i: usize| Point::new((i % w) as f64, (i / w) as f64);
    let extreme = |key: &dyn Fn(Point) -> f64| {
        component
            .iter()
            .map(|&i| point(i))
            .max_by(|a, b| key(*a).total_cmp(&key(*b)))
            .expect("non-empty component")
    };
    let mut quad = [
        extreme(&|p| -(p.x + p.y)),
        extreme(&|p| p.x - p.y),
        extreme(&|p| p.x + p.y),
        extreme(&|p| p.y - p.x),
    ];

    // Refine: fit a line to the boundary pixels along each side, intersect.
    let boundary: Vec<Point> = component
        .iter()
        .copied()
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            x == 0
                || y == 0
                || x + 1 == w
                || y + 1 == h
                || !in_component[i - 1]
                || !in_component[i + 1]
                || !in_component[i - w]
                || !in_component[i + w]
        })
        .map(point)
        .collect();
    let tolerance = 0.03 * ((w * w + h * h) as f64).sqrt();
    let lines: Vec<Option<(Point, Point)>> = (0..4)
        .map(|side| {
            let (a, b) = (quad[side], quad[(side + 1) % 4]);
            let near: Vec<Point> = boundary
                .iter()
                .copied()
                .filter(|&p| {
                    let (distance, t) = distance_to_segment(p, a, b);
                    distance < tolerance && (0.1..0.9).contains(&t)
                })
                .collect();
            line_through(&near)
        })
        .collect();
    if lines.iter().all(Option::is_some) {
        let refined: Option<Vec<Point>> = (0..4)
            .map(|corner| intersect(lines[(corner + 3) % 4]?, lines[corner]?))
            .collect();
        if let Some(refined) = refined {
            if refined
                .iter()
                .zip(quad.iter())
                .all(|(r, q)| (r.x - q.x).hypot(r.y - q.y) < 2.0 * tolerance)
            {
                quad.copy_from_slice(&refined);
            }
        }
    }

    let scale = factor as f64;
    let quad = quad.map(|p| Point::new((p.x + 0.5) * scale, (p.y + 0.5) * scale));
    let fraction = quad_area(&quad) / (width * height) as f64;
    let horizontal = (quad[1].x - quad[0].x).hypot(quad[1].y - quad[0].y)
        + (quad[2].x - quad[3].x).hypot(quad[2].y - quad[3].y);
    let vertical = (quad[3].x - quad[0].x).hypot(quad[3].y - quad[0].y)
        + (quad[2].x - quad[1].x).hypot(quad[2].y - quad[1].y);
    let aspect = horizontal / vertical;
    ((0.1..0.97).contains(&fraction) && (0.5..1.1).contains(&aspect)).then_some(quad)
}

/// Photo darkness at a reduced size, normalised by the local paper level
/// (a sliding maximum), so shadows do not read as print.
struct DarknessMap {
    plane: Plane,
    /// Summed-area table of `plane`, with a zero row and column in front.
    sums: Vec<f64>,
    scale: f64,
}

impl DarknessMap {
    fn new(photo: &RgbImage, long_side: usize) -> Self {
        let (width, height) = (photo.width() as usize, photo.height() as usize);
        // Sized by the short side as for a 4:3 photo, so that a portrait
        // camera frame (16:9) is not sampled more coarsely across the page.
        let factor = (width.min(height) as f64 * 4.0 / 3.0 / long_side as f64).max(1.0);
        let (w, h) = (
            ((width as f64 / factor).round() as usize).max(1),
            ((height as f64 / factor).round() as usize).max(1),
        );
        let mut value = Plane::new(w, h);
        let mut whiteness = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let [r, g, b] = sample_rgb(
                    photo,
                    (x as f64 + 0.5) * factor - 0.5,
                    (y as f64 + 0.5) * factor - 0.5,
                );
                value.data[y * w + x] = r.max(g).max(b);
                whiteness.data[y * w + x] = r.min(g).min(b);
            }
        }
        let paper = sliding_max(&value, (w.max(h) / 40).max(3));
        // Wide areas that are dark or coloured are the desk or whatever lies
        // around the sheet, not ink: their texture (wood grain, marble veins,
        // a bag) would otherwise pass for print. The sheet is white: its
        // weakest colour channel is still bright. Closing that removes thin
        // print (lines, letters, blue handwriting) and keeps those areas, with
        // a sharp edge at the sheet's.
        let reach = (w.min(h) / 50).max(2);
        let closed = sliding_min(&sliding_max(&whiteness, reach), reach);
        let mut levels = closed.data.clone();
        levels.sort_by(f32::total_cmp);
        let sheet = levels[levels.len() * 9 / 10].max(1.0);
        let mut plane = Plane::new(w, h);
        for i in 0..value.data.len() {
            let on_paper = ((closed.data[i] / sheet - 0.65) / 0.15).clamp(0.0, 1.0);
            plane.data[i] =
                on_paper * (1.0 - value.data[i] / paper.data[i].max(1.0)).clamp(0.0, 1.0);
        }
        let mut sums = vec![0.0_f64; (w + 1) * (h + 1)];
        for y in 0..h {
            for x in 0..w {
                sums[(y + 1) * (w + 1) + x + 1] = f64::from(plane.at(x, y))
                    + sums[y * (w + 1) + x + 1]
                    + sums[(y + 1) * (w + 1) + x]
                    - sums[y * (w + 1) + x];
            }
        }
        Self {
            plane,
            sums,
            scale: 1.0 / factor,
        }
    }

    /// Mean darkness of the square of side `size` (photo pixels) around `p`;
    /// outside the photo counts as blank. Sampling a template level coarser
    /// than this map at single points would miss thin lines between them.
    fn area(&self, p: Point, size: f64) -> f32 {
        let half = 0.5 * size * self.scale;
        if half < 0.75 {
            return self.sample(p);
        }
        let (cx, cy) = (p.x * self.scale, p.y * self.scale);
        let total = self.summed(cx + half, cy + half)
            - self.summed(cx - half, cy + half)
            - self.summed(cx + half, cy - half)
            + self.summed(cx - half, cy - half);
        (total / (4.0 * half * half)) as f32
    }

    /// Sum of the map over `[0, x) x [0, y)` (map pixels, fractional edges).
    fn summed(&self, x: f64, y: f64) -> f64 {
        let (w, h) = (self.plane.width, self.plane.height);
        let x = x.clamp(0.0, w as f64);
        let y = y.clamp(0.0, h as f64);
        let (x0, y0) = ((x as usize).min(w - 1), (y as usize).min(h - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let stride = w + 1;
        let at = |i: usize, j: usize| self.sums[j * stride + i];
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
        let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn sample(&self, p: Point) -> f32 {
        let (x, y) = (p.x * self.scale - 0.5, p.y * self.scale - 0.5);
        if x < 0.0
            || y < 0.0
            || x >= (self.plane.width - 1) as f64
            || y >= (self.plane.height - 1) as f64
        {
            return 0.0;
        }
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let top = self.plane.at(x0, y0) * (1.0 - fx) + self.plane.at(x0 + 1, y0) * fx;
        let bottom = self.plane.at(x0, y0 + 1) * (1.0 - fx) + self.plane.at(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// Mean over a (2 `radius` + 1)² window, clipped at the borders.
fn blur(plane: &Plane, radius: usize) -> Plane {
    if radius == 0 {
        return Plane {
            width: plane.width,
            height: plane.height,
            data: plane.data.clone(),
        };
    }
    let (w, h) = (plane.width, plane.height);
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            let sum: f32 = (y0..y1)
                .flat_map(|j| (x0..x1).map(move |i| (i, j)))
                .map(|(i, j)| plane.at(i, j))
                .sum();
            out.data[y * w + x] = sum / ((x1 - x0) * (y1 - y0)) as f32;
        }
    }
    out
}

fn sliding_max(plane: &Plane, radius: usize) -> Plane {
    sliding(plane, radius, 0.0, f32::max)
}

fn sliding_min(plane: &Plane, radius: usize) -> Plane {
    sliding(plane, radius, f32::MAX, f32::min)
}

/// `pick` over a (2 `radius` + 1)² window, clipped at the borders.
fn sliding(plane: &Plane, radius: usize, start: f32, pick: fn(f32, f32) -> f32) -> Plane {
    let (w, h) = (plane.width, plane.height);
    let mut rows = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            rows.data[y * w + x] = (x0..x1).map(|i| plane.at(i, y)).fold(start, pick);
        }
    }
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(h));
        for x in 0..w {
            out.data[y * w + x] = (y0..y1).map(|j| rows.at(x, j)).fold(start, pick);
        }
    }
    out
}

/// Correlation given up per unit of [`Registrar::distortion`].
const SHAPE_WEIGHT: f64 = 0.1;

/// Upright outlines of each orientation and size kept from the tiny level.
const SHORTLIST: usize = 16;
/// Of those, how many are also tried tilted.
const TILTED: usize = 6;

/// Contrast of a photo region, relative to the template's, below which it is
/// too faint to be the printed sheet.
const FAINT: f64 = 0.3;

/// Mean-centred template darkness at one coarse size.
struct TemplateLevel {
    /// Only the rows used for matching (the top of the page).
    plane: Plane,
    norm: f64,
    /// Whether each level pixel takes part in matching.
    kept: Vec<bool>,
    /// Template pixels per level pixel.
    step: f64,
    /// Level pixels each sample averages over (both on the template and the
    /// photo): more than one widens the score's peak, so that a coarse grid
    /// of starts does not step over it.
    spread: usize,
    /// Full template height in template pixels.
    page_height: f64,
}

impl TemplateLevel {
    fn new(
        template_gray: &image::GrayImage,
        width: usize,
        matched_fraction: f64,
        spread: usize,
        ignored: &[Rect],
    ) -> Self {
        let (tw, th) = (template_gray.width() as f64, template_gray.height() as f64);
        let step = tw / width as f64;
        let height = ((th * matched_fraction) / step).round() as usize;
        let mut plane = Plane::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Box average of the template pixels this level pixel covers.
                let (x0, y0) = ((x as f64 * step) as u32, (y as f64 * step) as u32);
                let x1 = (((x + 1) as f64 * step) as u32).clamp(x0 + 1, template_gray.width());
                let y1 = (((y + 1) as f64 * step) as u32).clamp(y0 + 1, template_gray.height());
                let mut sum = 0.0;
                for ty in y0..y1 {
                    for tx in x0..x1 {
                        sum += 1.0 - f32::from(template_gray.get_pixel(tx, ty)[0]) / 255.0;
                    }
                }
                plane.data[y * width + x] = sum / ((x1 - x0) * (y1 - y0)) as f32;
            }
        }
        let mut plane = blur(&plane, spread / 2);
        let kept: Vec<bool> = (0..plane.data.len())
            .map(|i| {
                let centre = Point::new(
                    ((i % width) as f64 + 0.5) * step,
                    ((i / width) as f64 + 0.5) * step,
                );
                !ignored.iter().any(|r| r.contains(centre))
            })
            .collect();
        let count = kept.iter().filter(|&&k| k).count().max(1);
        let mean = plane
            .data
            .iter()
            .zip(&kept)
            .filter(|(_, &k)| k)
            .map(|(v, _)| v)
            .sum::<f32>()
            / count as f32;
        for (v, &k) in plane.data.iter_mut().zip(&kept) {
            *v = if k { *v - mean } else { 0.0 };
        }
        let norm = plane
            .data
            .iter()
            .map(|&v| f64::from(v * v))
            .sum::<f64>()
            .sqrt();
        Self {
            plane,
            norm,
            kept,
            step,
            spread,
            page_height: th,
        }
    }

    /// Normalised cross-correlation with the photo when the template corners
    /// sit on `quad` (photo pixels).
    fn score(&self, darkness: &DarknessMap, quad: &[Point; 4]) -> f64 {
        let corners = Rect {
            x: 0.0,
            y: 0.0,
            width: self.plane.width as f64 * self.step,
            height: self.page_height,
        }
        .corners();
        let Some(map) = Homography::from_points(&corners, quad) else {
            return -1.0;
        };
        let (w, h) = (self.plane.width, self.plane.height);
        // Photo pixels one level pixel spans.
        let size =
            (quad[1].x - quad[0].x).hypot(quad[1].y - quad[0].y) / w as f64 * self.spread as f64;
        let mut samples = Vec::with_capacity(w * h);
        let mut templates = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                if !self.kept[y * w + x] {
                    continue;
                }
                let p = map.apply(Point::new(
                    (x as f64 + 0.5) * self.step,
                    (y as f64 + 0.5) * self.step,
                ));
                samples.push(darkness.area(p, size));
                templates.push(self.plane.data[y * w + x]);
            }
        }
        let mean = samples.iter().sum::<f32>() / samples.len().max(1) as f32;
        let (mut dot, mut norm) = (0.0_f64, 0.0_f64);
        for (t, s) in templates.iter().zip(&samples) {
            let centred = f64::from(s - mean);
            dot += f64::from(*t) * centred;
            norm += centred * centred;
        }
        if norm <= 0.0 {
            return -1.0;
        }
        // Normalised cross-correlation, except that a faint pattern (a desk's
        // grain or stripes, shading) is not scaled up to the strength of
        // print: below a fraction of the template's contrast it counts as
        // that fraction.
        dot / (self.norm * norm.sqrt().max(FAINT * self.norm))
    }
}

/// Finds the page outline by matching the blank template print against the
/// photo: a search over position, size and rotation plus the given candidate
/// outlines on a tiny template, then corner-by-corner refinement of the best
/// starts on progressively finer templates.
pub struct Registrar {
    tiny: TemplateLevel,
    coarse: TemplateLevel,
    fine: TemplateLevel,
    /// Template width / height.
    aspect: f64,
}

impl Registrar {
    /// Only the top of the sheet is matched: the institutional header, rules,
    /// `Soru-PÇ` table and the `SINAV SORULARI -- GRUP` line are laid out the
    /// same on every exam, while the question texts below differ.
    pub const MATCHED_FRACTION: f64 = 0.35;

    /// `ignored` (template pixels) is left out of matching: parts of the top
    /// that change between exams, i.e. the question columns of the table.
    pub fn new(template_gray: &image::GrayImage, ignored: &[Rect]) -> Self {
        let level = |width, spread| {
            TemplateLevel::new(
                template_gray,
                width,
                Self::MATCHED_FRACTION,
                spread,
                ignored,
            )
        };
        Self {
            tiny: level(40, 2),
            coarse: level(80, 1),
            fine: level(160, 1),
            aspect: f64::from(template_gray.width()) / f64::from(template_gray.height()),
        }
    }

    /// Follows a page found in an earlier camera frame: only refines from the
    /// previous outline instead of searching the whole frame, so it is several
    /// times faster than [`Registrar::register`]. The caller falls back to a
    /// full search when the score drops.
    pub fn track(&self, photo: &RgbImage, previous: &[Point; 4]) -> ([Point; 4], f64) {
        let coarse_map = DarknessMap::new(photo, 240);
        let fine_map = DarknessMap::new(photo, 480);
        // Between frames the sheet mostly moves: find the shift first, so
        // that corner-by-corner refinement does not bend the outline instead.
        let pixel = 1.0 / coarse_map.scale;
        let shifts: Vec<[Point; 4]> = (-6..=6)
            .flat_map(|i| (-6..=6).map(move |j| (i, j)))
            .map(|(i, j)| {
                let (dx, dy) = (f64::from(i) * 2.0 * pixel, f64::from(j) * 2.0 * pixel);
                previous.map(|p| Point::new(p.x + dx, p.y + dy))
            })
            .collect();
        let shifted = shifts
            .into_iter()
            .map(|quad| (self.coarse.score(&coarse_map, &quad), quad))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("at least one shift")
            .1;
        let start = self
            .refine(&self.coarse, &coarse_map, shifted, &[2.0, 1.0])
            .0;
        self.refine(&self.fine, &fine_map, start, &[2.0, 1.0, 0.5, 0.25])
    }

    /// Best page outline (photo pixels) and its correlation in [-1, 1].
    pub fn register(&self, photo: &RgbImage, candidates: &[[Point; 4]]) -> ([Point; 4], f64) {
        let (width, height) = (f64::from(photo.width()), f64::from(photo.height()));
        let tiny_map = DarknessMap::new(photo, 120);
        let coarse_map = DarknessMap::new(photo, 240);
        let fine_map = DarknessMap::new(photo, 480);

        // Upright pages first, in any of the four orientations (the sheet may
        // lie sideways in the frame). Fills above 1 are close-ups of the top
        // of the sheet (the matched part), e.g. the phone held over the
        // header and the Soru-PÇ table; the rest of the page then lies
        // outside the frame, on any side. The best few of every orientation
        // and size are also tried tilted by up to about 10 degrees (a tilt
        // lowers the upright score most at large sizes, so one best list over
        // all sizes would miss those).
        let sizes: Vec<(u32, f64)> = (0..4)
            // 10% apart, from a page well inside the frame to a close-up.
            .flat_map(|turn| (0..17).map(move |k| (turn, 0.62 * 1.1_f64.powi(k))))
            .collect();
        let searched = parallel_map(&sizes, |&(turn, fill)| {
            let (fw, fh) = if turn % 2 == 1 {
                (height, width)
            } else {
                (width, height)
            };
            let page_h = fill * fh.min(fw / self.aspect);
            let page_w = page_h * self.aspect;
            // The matched top of the page must be in view.
            let band = Self::MATCHED_FRACTION * page_h;
            let (top_min, top_max) = (-0.08 * fh, (fh - band).max(-0.08 * fh));
            let rows = ((top_max - top_min) / (0.04 * fh)).ceil().max(1.0) as i32;
            let mut found: Vec<(f64, [Point; 4])> = Vec::new();
            for j in 0..=rows {
                let top = top_min + (top_max - top_min) * f64::from(j) / f64::from(rows);
                for i in -4..=4 {
                    let center =
                        Point::new(fw / 2.0 + f64::from(i) * 0.04 * fw, top + page_h / 2.0);
                    let quad = page_quad(center, page_w, page_h, 0.0)
                        .map(|p| turn_back(p, turn, width, height));
                    found.push((self.tiny.score(&tiny_map, &quad), quad));
                }
            }
            found.sort_by(|a, b| b.0.total_cmp(&a.0));
            found.truncate(SHORTLIST);
            for (score, quad) in &mut found {
                *score = self.coarse.score(&coarse_map, quad);
            }
            found.sort_by(|a, b| b.0.total_cmp(&a.0));
            let tilted: Vec<(f64, [Point; 4])> = found
                .iter()
                .take(TILTED)
                .flat_map(|(_, quad)| {
                    [-9.0_f64, -6.0, -3.0, 3.0, 6.0, 9.0].map(|degrees| rotate_quad(quad, degrees))
                })
                .map(|quad| (self.coarse.score(&coarse_map, &quad), quad))
                .collect();
            found.extend(tilted);
            found
        });
        let mut scored: Vec<(f64, [Point; 4])> = searched.into_iter().flatten().collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        let starts = distinct(
            candidates
                .iter()
                .copied()
                .chain(scored.into_iter().map(|(_, quad)| quad)),
            0.05 * height,
            candidates.len() + 12,
        );
        let fitness = |found: &([Point; 4], f64)| found.1 - self.distortion(&found.0);
        let mut refined = parallel_map(&starts, |&start| {
            self.refine(&self.coarse, &coarse_map, start, &[8.0, 4.0, 2.0, 1.0])
        });
        refined.sort_by(|a, b| fitness(b).total_cmp(&fitness(a)));
        refined.truncate(3);
        parallel_map(&refined, |&(start, _)| {
            self.refine(&self.fine, &fine_map, start, &[2.0, 1.0, 0.5, 0.25])
        })
        .into_iter()
        .max_by(|a, b| fitness(a).total_cmp(&fitness(b)))
        .expect("at least one start")
    }

    /// How far `quad` is from the shape of a flat sheet seen from a phone:
    /// sides in the template's proportion to the top edge, a bottom edge as
    /// long as the top. Only the top of the sheet is matched, so without this
    /// the bottom corners may drift (bending the table's columns) at no cost.
    fn distortion(&self, quad: &[Point; 4]) -> f64 {
        let length = |a: Point, b: Point| (a.x - b.x).hypot(a.y - b.y);
        let top = length(quad[0], quad[1]).max(1.0);
        let off = |ratio: f64| (ratio - 1.0).abs();
        SHAPE_WEIGHT
            * (off(length(quad[0], quad[3]) * self.aspect / top)
                + off(length(quad[1], quad[2]) * self.aspect / top)
                + off(length(quad[3], quad[2]) / top))
    }

    /// Corner-by-corner hill climbing from `start`; returns the outline and
    /// its correlation.
    fn refine(
        &self,
        level: &TemplateLevel,
        darkness: &DarknessMap,
        start: [Point; 4],
        steps: &[f64],
    ) -> ([Point; 4], f64) {
        let fitness = |quad: &[Point; 4]| level.score(darkness, quad) - self.distortion(quad);
        let (mut best, mut best_score) = (start, fitness(&start));
        let pixel = 1.0 / darkness.scale;
        for &step in steps {
            let delta = step * pixel;
            let moves = [(delta, 0.0), (-delta, 0.0), (0.0, delta), (0.0, -delta)];
            for _ in 0..4 {
                let mut improved = false;
                // The whole outline: shifted, scaled or turned around its
                // centre (by about `delta` at its corners), which single
                // corner moves can only do through worse outlines.
                let size = (best[0].x - best[2].x)
                    .hypot(best[0].y - best[2].y)
                    .max(1.0);
                let (cx, cy) = (
                    best.iter().map(|p| p.x).sum::<f64>() / 4.0,
                    best.iter().map(|p| p.y).sum::<f64>() / 4.0,
                );
                let around = |scale: f64, angle: f64| {
                    let (sin, cos) = angle.sin_cos();
                    best.map(|p| {
                        let (x, y) = ((p.x - cx) * scale, (p.y - cy) * scale);
                        Point::new(cx + x * cos - y * sin, cy + x * sin + y * cos)
                    })
                };
                let turn = 2.0 * delta / size;
                let whole = moves
                    .iter()
                    .map(|&(dx, dy)| best.map(|p| Point::new(p.x + dx, p.y + dy)))
                    .chain([
                        around(1.0 + turn, 0.0),
                        around(1.0 - turn, 0.0),
                        around(1.0, turn),
                        around(1.0, -turn),
                    ])
                    .collect::<Vec<_>>();
                for candidate in whole {
                    let score = fitness(&candidate);
                    if score > best_score {
                        (best, best_score, improved) = (candidate, score, true);
                    }
                }
                for corner in 0..4 {
                    for (dx, dy) in moves {
                        let mut moved = best;
                        moved[corner] = Point::new(moved[corner].x + dx, moved[corner].y + dy);
                        let score = fitness(&moved);
                        if score > best_score {
                            (best, best_score, improved) = (moved, score, true);
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        let _ = best_score;
        (best, level.score(darkness, &best))
    }
}

/// Up to `count` of `quads`, in order, skipping any whose corners all lie
/// within `apart` pixels of one already taken.
fn distinct(
    quads: impl IntoIterator<Item = [Point; 4]>,
    apart: f64,
    count: usize,
) -> Vec<[Point; 4]> {
    let mut taken: Vec<[Point; 4]> = Vec::new();
    for quad in quads {
        if taken.len() == count {
            break;
        }
        let far = taken.iter().all(|t| {
            t.iter()
                .zip(&quad)
                .map(|(a, b)| (a.x - b.x).hypot(a.y - b.y))
                .fold(0.0, f64::max)
                > apart
        });
        if far {
            taken.push(quad);
        }
    }
    taken
}

/// `f` over `items`, spread over the CPU's cores, results in order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, items.len().max(1));
    let chunk = items.len().div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("registration thread"))
            .collect()
    })
}

/// Corners (top-left, top-right, bottom-right, bottom-left) of a `width` x
/// `height` page centred on `center` and turned `degrees` clockwise.
fn page_quad(center: Point, width: f64, height: f64, degrees: f64) -> [Point; 4] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(u, v): (f64, f64)| {
        let (x, y) = (u * width, v * height);
        Point::new(center.x + x * cos - y * sin, center.y + x * sin + y * cos)
    })
}

/// `quad` turned `degrees` clockwise around its centre.
fn rotate_quad(quad: &[Point; 4], degrees: f64) -> [Point; 4] {
    let cx = quad.iter().map(|p| p.x).sum::<f64>() / 4.0;
    let cy = quad.iter().map(|p| p.y).sum::<f64>() / 4.0;
    let (sin, cos) = degrees.to_radians().sin_cos();
    quad.map(|p| {
        let (x, y) = (p.x - cx, p.y - cy);
        Point::new(cx + x * cos - y * sin, cy + x * sin + y * cos)
    })
}

/// A point of the frame turned `turn` quarter turns counter-clockwise (where
/// the sheet looks upright) back into the `width` x `height` frame itself.
fn turn_back(p: Point, turn: u32, width: f64, height: f64) -> Point {
    match turn % 4 {
        0 => p,
        1 => Point::new(width - p.y, p.x),
        2 => Point::new(width - p.x, height - p.y),
        _ => Point::new(p.y, height - p.x),
    }
}

/// Template-space darkness of the printed sheet, used to align cells.
pub struct Aligner {
    template: Plane,
    /// How far (template pixels) a region may be shifted from where the page
    /// outline puts it.
    pub search_radius: usize,
    /// How far each cell edge may move to snap onto a printed line.
    pub edge_radius: usize,
}

impl Aligner {
    /// Aligns against the borders of `cells` (template pixels) drawn on an
    /// empty `width` x `height` sheet, so it depends only on the layout and
    /// not on the exam's printed text.
    pub fn from_cells(width: usize, height: usize, cells: &[Rect]) -> Self {
        let mut template = Plane::new(width, height);
        let mut mark = |x: isize, y: isize| {
            if (0..width as isize).contains(&x) && (0..height as isize).contains(&y) {
                template.data[y as usize * width + x as usize] = 1.0;
            }
        };
        for cell in cells {
            let (x0, y0) = (cell.x.round() as isize, cell.y.round() as isize);
            let (x1, y1) = (
                (cell.x + cell.width).round() as isize,
                (cell.y + cell.height).round() as isize,
            );
            for x in x0..=x1 {
                mark(x, y0);
                mark(x, y1);
            }
            for y in y0..=y1 {
                mark(x0, y);
                mark(x1, y);
            }
        }
        Self {
            template,
            // Registration already places the page within a few pixels; a
            // radius under half a table row cannot lock onto the wrong row.
            search_radius: (height / 120).max(3),
            edge_radius: (height / 200).max(2),
        }
    }

    pub fn template_size(&self) -> (usize, usize) {
        (self.template.width, self.template.height)
    }

    /// Photo darkness of `window` (template pixels), seen through `to_photo`.
    fn sample(&self, photo: &RgbImage, to_photo: &Homography, window: Rect) -> Plane {
        let (w, h) = (
            window.width.round() as usize,
            window.height.round() as usize,
        );
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = to_photo.apply(Point::new(
                    window.x + x as f64 + 0.5,
                    window.y + y as f64 + 0.5,
                ));
                let [r, g, b] = sample_rgb(photo, p.x - 0.5, p.y - 0.5);
                plane.data[y * w + x] = r.max(g).max(b);
            }
        }
        // Paper level of this window, robust to shadows across the page.
        let mut sorted = plane.data.clone();
        sorted.sort_by(f32::total_cmp);
        let paper = sorted[(sorted.len() * 9) / 10].max(1.0);
        for value in &mut plane.data {
            *value = (1.0 - *value / paper).clamp(0.0, 1.0);
        }
        plane
    }

    /// Aligns `cells` (template pixels) that share one printed region and
    /// returns them in template pixels, corrected for the photo.
    pub fn align(&self, photo: &RgbImage, to_photo: &Homography, cells: &[Rect]) -> Vec<Rect> {
        let Some(first) = cells.first() else {
            return Vec::new();
        };
        let margin = 6.0;
        let region = cells.iter().skip(1).fold(*first, |acc, c| acc.union(c));
        let region = Rect {
            x: region.x - margin,
            y: region.y - margin,
            width: region.width + 2.0 * margin,
            height: region.height + 2.0 * margin,
        };
        let r = self.search_radius as f64;
        let window = Rect {
            x: region.x - r,
            y: region.y - r,
            width: region.width + 2.0 * r,
            height: region.height + 2.0 * r,
        };
        let photo_dark = self.sample(photo, to_photo, window);

        // Region shift from row and column darkness profiles.
        let (rx, ry) = (region.x.round() as isize, region.y.round() as isize);
        let (rw, rh) = (
            region.width.round() as isize,
            region.height.round() as isize,
        );
        let radius = self.search_radius as isize;
        let template_rows: Vec<f32> = (0..rh)
            .map(|y| self.template_row(ry + y, rx, rx + rw))
            .collect();
        let template_cols: Vec<f32> = (0..rw)
            .map(|x| self.template_col(rx + x, ry, ry + rh))
            .collect();
        let best_shift = |template: &[f32], photo_profile: &dyn Fn(isize) -> f32| {
            (-radius..=radius)
                .map(|shift| {
                    let score: f32 = template
                        .iter()
                        .enumerate()
                        .map(|(i, &t)| t * photo_profile(i as isize + radius + shift))
                        .sum();
                    (shift, score)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.abs().cmp(&a.0.abs())))
                .map(|(shift, _)| shift)
                .unwrap_or(0)
        };
        let dy = best_shift(&template_rows, &|y| {
            photo_dark.row_mean(y, radius, radius + rw)
        });
        let dx = best_shift(&template_cols, &|x| {
            photo_dark.column_mean(x, radius, radius + rh)
        });

        // Snap each cell edge to the strongest nearby printed line.
        let edge = self.edge_radius as isize;
        let local = |p: Point| (p.x - window.x, p.y - window.y);
        cells
            .iter()
            .map(|cell| {
                let shifted = cell.translated(dx as f64, dy as f64);
                let (x0, y0) = local(Point::new(shifted.x, shifted.y));
                let (x1, y1) = (x0 + shifted.width, y0 + shifted.height);
                let inset_x = shifted.width * 0.15;
                let inset_y = shifted.height * 0.15;
                let snap_h = |y: f64| {
                    snap(y, edge, |c| {
                        photo_dark.row_mean(c, (x0 + inset_x) as isize, (x1 - inset_x) as isize)
                    })
                };
                let snap_v = |x: f64| {
                    snap(x, edge, |c| {
                        photo_dark.column_mean(c, (y0 + inset_y) as isize, (y1 - inset_y) as isize)
                    })
                };
                let (top, bottom, left, right) = (snap_h(y0), snap_h(y1), snap_v(x0), snap_v(x1));
                if bottom - top < shifted.height * 0.6 || right - left < shifted.width * 0.6 {
                    return shifted;
                }
                Rect {
                    x: left + window.x,
                    y: top + window.y,
                    width: right - left,
                    height: bottom - top,
                }
            })
            .collect()
    }

    fn template_row(&self, y: isize, x0: isize, x1: isize) -> f32 {
        self.template.row_mean(y, x0, x1)
    }

    fn template_col(&self, x: isize, y0: isize, y1: isize) -> f32 {
        self.template.column_mean(x, y0, y1)
    }
}

/// Position within `radius` of `at` with the darkest line, if it is clearly a line.
fn snap(at: f64, radius: isize, strength: impl Fn(isize) -> f32) -> f64 {
    let center = at.round() as isize;
    let (best, value) = (center - radius..=center + radius)
        .map(|c| (c, strength(c)))
        .max_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then((b.0 - center).abs().cmp(&(a.0 - center).abs()))
        })
        .unwrap_or((center, 0.0));
    if value >= 0.25 {
        best as f64 + 0.5
    } else {
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_control_points_and_lines() {
        let from = Rect {
            x: 0.0,
            y: 0.0,
            width: 850.0,
            height: 1100.0,
        }
        .corners();
        let to = [
            Point::new(103.0, 87.0),
            Point::new(1050.0, 120.0),
            Point::new(1101.0, 1490.0),
            Point::new(60.0, 1452.0),
        ];
        let h = Homography::from_points(&from, &to).unwrap();
        for (a, b) in from.iter().zip(to.iter()) {
            let p = h.apply(*a);
            assert!((p.x - b.x).abs() < 1e-6 && (p.y - b.y).abs() < 1e-6);
        }
        assert!(Homography::from_points(&[Point::new(0.0, 0.0); 4], &to).is_none());
    }

    #[test]
    fn otsu_splits_two_populations() {
        let values: Vec<f32> = (0..100)
            .map(|i| if i < 50 { 40.0 } else { 200.0 })
            .collect();
        let t = otsu(&values);
        assert!(t > 40.0 && t < 200.0);
    }

    fn draw_page(width: u32, height: u32, quad: &[Point; 4]) -> RgbImage {
        let inside = |x: f64, y: f64| {
            (0..4).all(|i| {
                let (a, b) = (quad[i], quad[(i + 1) % 4]);
                (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x) >= 0.0
            })
        };
        RgbImage::from_fn(width, height, |x, y| {
            if inside(x as f64 + 0.5, y as f64 + 0.5) {
                image::Rgb([235, 233, 228])
            } else {
                image::Rgb([120, 82, 50])
            }
        })
    }

    #[test]
    fn page_corners_are_found_on_a_contrasting_desk() {
        let quad = [
            Point::new(130.0, 95.0),
            Point::new(890.0, 140.0),
            Point::new(930.0, 1180.0),
            Point::new(95.0, 1150.0),
        ];
        let found = detect_page(&draw_page(1000, 1300, &quad)).unwrap();
        for (f, q) in found.iter().zip(quad.iter()) {
            assert!((f.x - q.x).hypot(f.y - q.y) < 4.0, "{f:?} vs {q:?}");
        }
    }

    #[test]
    fn a_frame_without_a_page_outline_is_rejected() {
        let blank = RgbImage::from_pixel(600, 800, image::Rgb([240, 240, 240]));
        assert!(detect_page(&blank).is_none());
    }

    #[test]
    fn warp_quad_extracts_an_upright_patch() {
        let photo = RgbImage::from_fn(200, 100, |x, _| image::Rgb([(x % 256) as u8, 0, 0]));
        let quad = Rect {
            x: 50.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        }
        .corners();
        let patch = warp_quad(&photo, &quad, 100, 50);
        assert_eq!(patch.dimensions(), (100, 50));
        assert!((i32::from(patch.get_pixel(0, 10)[0]) - 50).abs() <= 1);
        assert!((i32::from(patch.get_pixel(99, 10)[0]) - 149).abs() <= 1);
    }
}
