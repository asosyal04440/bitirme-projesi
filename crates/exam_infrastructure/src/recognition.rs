//! On-device reader for handwritten digit cells (scores, the `Not:` total and
//! the student number), running the ONNX model trained by `tools/recognizer`.
//!
//! Preprocessing and decoding mirror `tools/recognizer/model.py`; the parity
//! test (`tests/recognizer_parity.rs`) checks both against outputs of the
//! Python reference.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::dims;

/// Classes after the CTC blank. `.` is any decimal separator; `~` means a
/// crossed-out correction is present in the cell.
pub const ALPHABET: &str = "0123456789.~";
pub const INPUT_HEIGHT: usize = 32;
pub const MIN_WIDTH: usize = 16;
pub const MAX_WIDTH: usize = 512;
const WIDTH_MULTIPLE: usize = 4;
/// Input columns per output frame (two 2x poolings along the width).
const FRAME_STRIDE: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct RgbImage<'a> {
    width: usize,
    height: usize,
    pixels: &'a [u8],
}

impl<'a> RgbImage<'a> {
    /// `pixels` are row-major RGB bytes, `width * height * 3` long.
    pub fn new(width: usize, height: usize, pixels: &'a [u8]) -> Result<Self, RecognitionError> {
        if width == 0 || height == 0 {
            return Err(RecognitionError::InvalidImage("image is empty"));
        }
        if pixels.len() != width * height * 3 {
            return Err(RecognitionError::InvalidImage(
                "pixel buffer does not match width * height * 3",
            ));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    fn channel(&self, x: usize, y: usize, c: usize) -> f32 {
        f32::from(self.pixels[(y * self.width + x) * 3 + c])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellReading {
    /// Decoded text, e.g. `17`, `7.5`, `~14` (corrected to 14) or empty (blank cell).
    pub text: String,
    /// Minimum over frames of the winning class probability.
    pub confidence: f32,
}

impl CellReading {
    pub fn is_blank(&self) -> bool {
        self.text.is_empty()
    }

    pub fn has_correction(&self) -> bool {
        self.text.contains('~')
    }

    /// The text without correction markers, for parsing as a score or number.
    pub fn value_text(&self) -> String {
        self.text.replace('~', "")
    }
}

#[derive(Debug)]
pub enum RecognitionError {
    InvalidImage(&'static str),
    Model(String),
}

impl fmt::Display for RecognitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImage(reason) => write!(f, "invalid cell image: {reason}"),
            Self::Model(reason) => write!(f, "recognizer model error: {reason}"),
        }
    }
}

impl std::error::Error for RecognitionError {}

impl From<TractError> for RecognitionError {
    fn from(error: TractError) -> Self {
        Self::Model(format!("{error:#}"))
    }
}

/// A preprocessed crop: 3 x INPUT_HEIGHT x `width`, values in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct CellTensor {
    pub width: usize,
    pub data: Vec<f32>,
}

fn target_width(height: usize, width: usize) -> usize {
    // Round half up, exactly like `math.floor(x + 0.5)` in model.py.
    let scaled = (width as f64 * INPUT_HEIGHT as f64 / height as f64 + 0.5).floor() as usize;
    scaled.clamp(MIN_WIDTH, MAX_WIDTH)
}

/// Pixel-centre aligned bilinear resize with edge clamping, then white padding
/// on the right to a multiple of 4 and scaling to [-1, 1] (channel-major).
pub fn preprocess(image: &RgbImage<'_>) -> CellTensor {
    let width = target_width(image.height, image.width);
    let padded = width.div_ceil(WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
    let source = |position: usize, target: usize, size: usize| {
        let coordinate = ((position as f64 + 0.5) * size as f64 / target as f64 - 0.5)
            .clamp(0.0, (size - 1) as f64);
        let low = coordinate.floor() as usize;
        (
            low,
            (low + 1).min(size - 1),
            (coordinate - low as f64) as f32,
        )
    };
    let rows: Vec<_> = (0..INPUT_HEIGHT)
        .map(|y| source(y, INPUT_HEIGHT, image.height))
        .collect();
    let columns: Vec<_> = (0..width).map(|x| source(x, width, image.width)).collect();

    let mut data = vec![1.0_f32; 3 * INPUT_HEIGHT * padded];
    for c in 0..3 {
        for (y, &(y0, y1, wy)) in rows.iter().enumerate() {
            for (x, &(x0, x1, wx)) in columns.iter().enumerate() {
                let top = image.channel(x0, y0, c) * (1.0 - wx) + image.channel(x1, y0, c) * wx;
                let bottom = image.channel(x0, y1, c) * (1.0 - wx) + image.channel(x1, y1, c) * wx;
                let value = top * (1.0 - wy) + bottom * wy;
                data[(c * INPUT_HEIGHT + y) * padded + x] = value / 127.5 - 1.0;
            }
        }
    }
    CellTensor {
        width: padded,
        data,
    }
}

/// Which field a crop comes from; it limits the characters decoding may emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    Score,
    DeclaredTotal,
    /// Digits only: a student number never has a decimal point or a correction.
    StudentNumber,
}

impl CellKind {
    pub fn allowed(self) -> &'static str {
        match self {
            Self::Score | Self::DeclaredTotal => ALPHABET,
            Self::StudentNumber => "0123456789",
        }
    }
}

/// Greedy CTC decoding of a (frames x classes) probability matrix over the
/// blank and the `allowed` characters; other classes are ignored, not
/// renormalised, so probability spent on an impossible character lowers the
/// confidence.
pub fn decode(probabilities: &[f32], classes: usize, allowed: &str) -> CellReading {
    let alphabet: Vec<char> = ALPHABET.chars().collect();
    let usable: Vec<bool> = (0..classes)
        .map(|class| {
            class == 0
                || alphabet
                    .get(class - 1)
                    .is_some_and(|c| allowed.contains(*c))
        })
        .collect();
    let mut text = String::new();
    let mut confidence = if probabilities.is_empty() {
        0.0
    } else {
        1.0_f32
    };
    let mut previous = 0;
    for frame in probabilities.chunks_exact(classes) {
        let (best, probability) = frame
            .iter()
            .copied()
            .enumerate()
            .filter(|&(class, _)| usable[class])
            .fold(
                (0, f32::MIN),
                |acc, item| if item.1 > acc.1 { item } else { acc },
            );
        confidence = confidence.min(probability);
        if best != 0 && best != previous {
            text.push(alphabet[best - 1]);
        }
        previous = best;
    }
    CellReading { text, confidence }
}

pub struct CellRecognizer {
    plan: Arc<TypedRunnableModel>,
}

impl CellRecognizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RecognitionError> {
        Self::from_model(tract_onnx::onnx().model_for_path(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RecognitionError> {
        Self::from_model(tract_onnx::onnx().model_for_read(&mut &*bytes)?)
    }

    fn from_model(model: InferenceModel) -> Result<Self, RecognitionError> {
        // The width stays symbolic, so one optimized plan serves every crop size.
        let width = model.sym("W");
        let plan = model
            .with_input_fact(0, f32::fact(dims!(1, 3, INPUT_HEIGHT, width)).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan })
    }

    pub fn read(
        &self,
        image: &RgbImage<'_>,
        kind: CellKind,
    ) -> Result<CellReading, RecognitionError> {
        let tensor = preprocess(image);
        let input = Tensor::from_shape(&[1, 3, INPUT_HEIGHT, tensor.width], &tensor.data)?;
        let outputs = self.plan.run(tvec!(input.into()))?;
        let output = &outputs[0];
        let shape = output.shape();
        if shape.len() != 3 || shape[1] != tensor.width / FRAME_STRIDE {
            return Err(RecognitionError::Model(format!(
                "unexpected output shape {shape:?}"
            )));
        }
        let probabilities = output.try_as_plain_ram()?.as_slice::<f32>()?;
        Ok(decode(probabilities, shape[2], kind.allowed()))
    }
}
