//! Diagnostics for field tests: what the reader saw in the last photo or
//! camera frame, drawn over the picture, plus a text report. A tester shares
//! them from the app when a sheet is not read as expected.

use exam_infrastructure::sheet::{SheetReading, MIN_REGISTRATION_SCORE};
use exam_infrastructure::vision::Point;
use image::{Rgb, RgbImage};
use std::fmt::Write;

/// The last picture the reader looked at and what it found there.
pub(crate) struct Seen {
    pub image: RgbImage,
    /// `"fotoğraf"` or `"kamera"`.
    pub source: &'static str,
    pub page_quad: Option<[Point; 4]>,
    pub registration_score: f64,
    pub table: Option<[Point; 4]>,
    pub reading: Option<SheetReading>,
}

const BLUE: Rgb<u8> = Rgb([37, 99, 235]);
const GREEN: Rgb<u8> = Rgb([22, 163, 74]);
const AMBER: Rgb<u8> = Rgb([217, 119, 6]);
const RED: Rgb<u8> = Rgb([220, 38, 38]);

/// The picture (at most 1600 px on the long side) with the page outline in
/// blue, the Soru-PÇ table in green and every read cell coloured by its
/// confidence: green ≥ 0.9, amber ≥ 0.6, red below.
pub(crate) fn annotated(seen: &Seen) -> RgbImage {
    let (w, h) = seen.image.dimensions();
    let scale = (1600.0 / f64::from(w.max(h))).min(1.0);
    let mut image = if scale < 1.0 {
        image::imageops::resize(
            &seen.image,
            (f64::from(w) * scale) as u32,
            (f64::from(h) * scale) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        seen.image.clone()
    };
    let scaled = |quad: &[Point; 4]| quad.map(|p| Point::new(p.x * scale, p.y * scale));
    if let Some(quad) = &seen.page_quad {
        draw_quad(&mut image, &scaled(quad), BLUE, 3);
    }
    if let Some(quad) = &seen.table {
        draw_quad(&mut image, &scaled(quad), GREEN, 3);
    }
    for field in seen.reading.iter().flat_map(|r| &r.fields) {
        if let Some(reading) = &field.reading {
            let colour = match reading.confidence {
                c if c >= 0.9 => GREEN,
                c if c >= 0.6 => AMBER,
                _ => RED,
            };
            draw_quad(&mut image, &scaled(&field.quad), colour, 2);
        }
    }
    image
}

/// A plain-text account of the reading, for the person debugging it.
pub(crate) fn report(seen: &Seen, model: &str) -> String {
    let mut text = String::new();
    let (w, h) = seen.image.dimensions();
    let _ = writeln!(text, "Sınav Okuyucu tanı raporu");
    let _ = writeln!(text, "Kaynak: {} ({w}x{h} px), model {model}", seen.source);
    let _ = writeln!(
        text,
        "Kâğıt bulundu: {} (eşleşme {:.2}, eşik {MIN_REGISTRATION_SCORE:.2})",
        if seen.registration_score >= MIN_REGISTRATION_SCORE {
            "evet"
        } else {
            "hayır"
        },
        seen.registration_score
    );
    for field in seen.reading.iter().flat_map(|r| &r.fields) {
        if let Some(reading) = &field.reading {
            let _ = writeln!(
                text,
                "{}: \"{}\" güven {:.2}",
                field.field_key, reading.text, reading.confidence
            );
        }
    }
    text
}

fn draw_quad(image: &mut RgbImage, quad: &[Point; 4], colour: Rgb<u8>, thickness: i64) {
    for i in 0..4 {
        draw_line(image, quad[i], quad[(i + 1) % 4], colour, thickness);
    }
}

fn draw_line(image: &mut RgbImage, a: Point, b: Point, colour: Rgb<u8>, thickness: i64) {
    let steps = (b.x - a.x).abs().max((b.y - a.y).abs()).ceil().max(1.0) as i64;
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    for step in 0..=steps {
        let t = step as f64 / steps as f64;
        let (x, y) = (
            (a.x + (b.x - a.x) * t).round() as i64,
            (a.y + (b.y - a.y) * t).round() as i64,
        );
        for dy in -thickness / 2..=thickness / 2 {
            for dx in -thickness / 2..=thickness / 2 {
                let (px, py) = (x + dx, y + dy);
                if (0..w).contains(&px) && (0..h).contains(&py) {
                    image.put_pixel(px as u32, py as u32, colour);
                }
            }
        }
    }
}
