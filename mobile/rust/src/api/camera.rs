//! Camera frames handed over from the live preview, and what scanning one
//! reports back to the screen.

use crate::api::reader::SheetResult;

/// Pixel layout of a camera frame as the platform delivers it.
pub enum FrameFormat {
    /// Android: Y plane plus U and V planes (planar or interleaved, see
    /// `pixel_stride`).
    Yuv420,
    /// iOS: one plane, 4 bytes per pixel in B, G, R, A order.
    Bgra8888,
    /// One plane, 4 bytes per pixel in R, G, B, A order (tests, still images).
    Rgba8888,
}

pub struct FramePlane {
    pub bytes: Vec<u8>,
    /// Bytes from the start of one row to the next.
    pub row_stride: u32,
    /// Bytes from one pixel to the next within a row.
    pub pixel_stride: u32,
}

pub struct CameraFrame {
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    pub planes: Vec<FramePlane>,
    /// Clockwise rotation (0, 90, 180 or 270) that turns the sensor image
    /// upright as the preview shows it.
    pub rotation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    /// No sheet in view.
    Searching,
    /// A sheet is in view and being read; frames are combined until the
    /// reading is certain.
    Reading,
    /// The sheet in view was read and saved just now (`sheet` is set).
    Saved,
    /// The sheet in view is the one saved last; show the next one.
    AlreadySaved,
}

pub struct ScanUpdate {
    pub state: ScanState,
    /// Corners (TL, TR, BR, BL) of the Soru-PÇ table as `[x0, y0, … x3, y3]`,
    /// normalised to the upright frame; empty when no sheet is in view.
    pub table: Vec<f32>,
    /// The saved sheet when `state` is `Saved`.
    pub sheet: Option<SheetResult>,
    /// True when the saved sheet was read with certainty (every value
    /// accepted and consistent); false when it was saved after waiting and
    /// needs the instructor's check.
    pub verified: bool,
    /// Frames combined into the current reading.
    pub frames: u32,
    /// Processing time of this frame.
    pub millis: u32,
}
