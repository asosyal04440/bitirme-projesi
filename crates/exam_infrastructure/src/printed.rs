//! Reading printed text on the sheet (course, exam title, the Soru-PÇ table
//! labels) with PaddleOCR's PP-OCRv4 mobile recognition model (Apache-2.0).
//!
//! Handwriting is read by our own digit model; printed text is what PaddleOCR
//! was trained for. The model reads one line of text from a crop that is 48 px
//! high. Its width must be fixed for `tract` (the model reshapes by the
//! width), so crops are padded to one of a few widths.

use crate::recognition::RecognitionError;
use image::RgbImage;
use std::sync::{Arc, OnceLock};
use tract_onnx::prelude::*;

const HEIGHT: u32 = 48;
/// Input widths a plan is built for; a crop uses the smallest that fits and is
/// squeezed into the largest when it is wider.
const WIDTHS: [u32; 2] = [320, 640];

#[derive(Debug, Clone, PartialEq)]
pub struct PrintedText {
    pub text: String,
    /// Minimum over frames of the winning class probability.
    pub confidence: f32,
}

pub struct PrintedTextReader {
    model: Vec<u8>,
    /// Output classes: CTC blank, the model's characters, then a space.
    classes: Vec<String>,
    plans: [OnceLock<Arc<TypedRunnableModel>>; WIDTHS.len()],
}

impl PrintedTextReader {
    /// `model` is the ONNX file, `keys` its character list, one per line.
    /// Plans are built on first use of each width.
    pub fn from_bytes(model: Vec<u8>, keys: &str) -> Result<Self, RecognitionError> {
        let characters: Vec<String> = keys.lines().map(str::to_owned).collect();
        if characters.is_empty() {
            return Err(RecognitionError::Model("empty character list".to_owned()));
        }
        let mut classes = Vec::with_capacity(characters.len() + 2);
        classes.push(String::new());
        classes.extend(characters);
        classes.push(" ".to_owned());
        Ok(Self {
            model,
            classes,
            plans: Default::default(),
        })
    }

    fn plan(&self, index: usize) -> Result<Arc<TypedRunnableModel>, RecognitionError> {
        if let Some(plan) = self.plans[index].get() {
            return Ok(plan.clone());
        }
        let width = WIDTHS[index] as usize;
        let plan = tract_onnx::onnx()
            .model_for_read(&mut &*self.model)?
            .with_input_fact(0, f32::fact([1, 3, HEIGHT as usize, width]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(self.plans[index].get_or_init(|| plan).clone())
    }

    /// Builds the model plans for every width now instead of on first use.
    pub fn warm_up(&self) -> Result<(), RecognitionError> {
        for index in 0..WIDTHS.len() {
            self.plan(index)?;
        }
        Ok(())
    }

    /// Reads the single line of printed text in `crop`.
    pub fn read(&self, crop: &RgbImage) -> Result<PrintedText, RecognitionError> {
        if crop.width() == 0 || crop.height() == 0 {
            return Err(RecognitionError::InvalidImage("empty crop"));
        }
        let wanted = (f64::from(crop.width()) * f64::from(HEIGHT) / f64::from(crop.height()))
            .ceil()
            .max(1.0) as u32;
        let index = WIDTHS
            .iter()
            .position(|&w| wanted <= w)
            .unwrap_or(WIDTHS.len() - 1);
        let plan_width = WIDTHS[index];
        let width = wanted.min(plan_width);
        let resized =
            image::imageops::resize(crop, width, HEIGHT, image::imageops::FilterType::Triangle);
        // BGR channel order (the model was trained on OpenCV images), values
        // in [-1, 1]; the padding is 0, as in PaddleOCR.
        let (w, h) = (plan_width as usize, HEIGHT as usize);
        let mut data = vec![0.0_f32; 3 * h * w];
        for (x, y, pixel) in resized.enumerate_pixels() {
            for (channel, &value) in [pixel[2], pixel[1], pixel[0]].iter().enumerate() {
                data[(channel * h + y as usize) * w + x as usize] = f32::from(value) / 127.5 - 1.0;
            }
        }
        let input = Tensor::from_shape(&[1, 3, h, w], &data)?;
        let outputs = self.plan(index)?.run(tvec!(input.into()))?;
        let output = &outputs[0];
        let shape = output.shape();
        if shape.len() != 3 || shape[2] != self.classes.len() {
            return Err(RecognitionError::Model(format!(
                "unexpected output shape {shape:?}"
            )));
        }
        let probabilities = output.try_as_plain_ram()?.as_slice::<f32>()?;
        // Only the frames covering the crop, not the padding.
        let frames = ((width as usize * shape[1]).div_ceil(w)).min(shape[1]);
        let mut text = String::new();
        let mut confidence = 1.0_f32;
        let mut previous = 0;
        for frame in probabilities.chunks_exact(shape[2]).take(frames) {
            let (best, probability) = frame
                .iter()
                .copied()
                .enumerate()
                .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
            confidence = confidence.min(probability);
            if best != 0 && best != previous {
                text.push_str(&self.classes[best]);
            }
            previous = best;
        }
        Ok(PrintedText {
            text: text.trim().to_owned(),
            confidence,
        })
    }
}
