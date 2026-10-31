//! Test recordings of live scanning on real sheets.
//!
//! Measured like video document recognition (MIDV-500, Arlazarov et al.
//! 2019): while recording, every camera frame is kept, upright, with the scan
//! state it led to and the sheet saved on it. Once the instructor has checked
//! the sheets, their confirmed values are the truth; the recording is packed
//! with them and the exam's definition, and replayed on a computer
//! (`live_eval::live_replay`) to measure a change on real handwriting.

use crate::api::camera::{ScanState, ScanUpdate};
use anyhow::Context;
use image::RgbImage;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Long side frames are kept at.
const KEPT_SIZE: u32 = 1280;

pub(crate) struct Recording {
    dir: PathBuf,
    next: u32,
}

impl Recording {
    pub(crate) fn start(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        let next = std::fs::read_dir(dir)?
            .filter(|e| {
                e.as_ref()
                    .is_ok_and(|e| e.path().extension().is_some_and(|x| x == "jpg"))
            })
            .count() as u32;
        Ok(Self {
            dir: dir.to_path_buf(),
            next,
        })
    }

    /// Keeps `rgb` and what scanning it led to.
    pub(crate) fn add(&mut self, rgb: &RgbImage, update: &ScanUpdate) -> anyhow::Result<()> {
        let (w, h) = rgb.dimensions();
        let scale = (f64::from(KEPT_SIZE) / f64::from(w.max(h))).min(1.0);
        let kept = if scale < 1.0 {
            image::imageops::resize(
                rgb,
                (f64::from(w) * scale) as u32,
                (f64::from(h) * scale) as u32,
                image::imageops::FilterType::Triangle,
            )
        } else {
            rgb.clone()
        };
        let name = format!("{:05}.jpg", self.next);
        let mut file = std::io::BufWriter::new(std::fs::File::create(self.dir.join(&name))?);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, 90).encode_image(&kept)?;
        let state = match update.state {
            ScanState::Searching => "searching",
            ScanState::Reading => "reading",
            ScanState::Saved => "saved",
            ScanState::AlreadySaved => "already_saved",
        };
        let sheet = update
            .sheet
            .as_ref()
            .map_or("null".to_owned(), |s| format!("\"{}\"", s.sheet_id));
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("frames.jsonl"))?;
        writeln!(
            log,
            "{{\"frame\": \"{name}\", \"state\": \"{state}\", \"sheet\": {sheet}}}"
        )?;
        self.next += 1;
        Ok(())
    }
}

/// `dir` and the `extra` files (name, content) in one zip next to it;
/// returns the zip's path. An extra file replaces one of the same name in
/// `dir`. JPEGs are stored as they are.
pub(crate) fn pack(dir: &Path, extra: &[(String, Vec<u8>)]) -> anyhow::Result<PathBuf> {
    let target = dir.with_extension("zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&target)?);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("file name")?;
        if extra.iter().any(|(extra, _)| extra == name) {
            continue;
        }
        zip.start_file(name, options)?;
        zip.write_all(&std::fs::read(&path)?)?;
    }
    for (name, content) in extra {
        zip.start_file(name.as_str(), options)?;
        zip.write_all(content)?;
    }
    zip.finish()?;
    Ok(target)
}
