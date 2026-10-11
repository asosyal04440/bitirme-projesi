//! The Rust recognizer must read cells exactly like the Python reference that
//! trained and evaluated the model (`tools/recognizer/export_fixtures.py`).

use exam_infrastructure::recognition::{
    decode, preprocess, CellKind, CellRecognizer, RgbImage, ALPHABET,
};
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Case {
    file: String,
    width: usize,
    height: usize,
    kind: CellKind,
    tensor_width: usize,
    tensor_sum: f64,
    text: String,
    confidence: f32,
}

fn cases() -> Vec<Case> {
    let expected =
        fs::read_to_string(root().join("tests/fixtures/recognizer/expected.tsv")).unwrap();
    expected
        .lines()
        .skip(1)
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            Case {
                file: fields[0].to_owned(),
                width: fields[1].parse().unwrap(),
                height: fields[2].parse().unwrap(),
                kind: match fields[3] {
                    "score" => CellKind::Score,
                    "total" => CellKind::DeclaredTotal,
                    "number" => CellKind::StudentNumber,
                    other => panic!("unknown kind {other}"),
                },
                tensor_width: fields[5].parse().unwrap(),
                tensor_sum: fields[6].parse().unwrap(),
                text: fields[7].to_owned(),
                confidence: fields[8].parse().unwrap(),
            }
        })
        .collect()
}

fn pixels(case: &Case) -> Vec<u8> {
    fs::read(root().join("tests/fixtures/recognizer").join(&case.file)).unwrap()
}

#[test]
fn preprocessing_matches_the_python_reference() {
    for case in cases() {
        let pixels = pixels(&case);
        let tensor = preprocess(&RgbImage::new(case.width, case.height, &pixels).unwrap());
        assert_eq!(tensor.width, case.tensor_width, "{}", case.file);
        let sum: f64 = tensor.data.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (sum - case.tensor_sum).abs() < 0.05,
            "{}: sum {sum} vs {}",
            case.file,
            case.tensor_sum
        );
    }
}

#[test]
fn bundled_model_reads_cells_like_the_python_reference() {
    let recognizer = CellRecognizer::load(root().join("../../models/digit-crnn-v1.onnx")).unwrap();
    let cases = cases();
    assert!(cases.len() >= 10);
    for case in &cases {
        let pixels = pixels(case);
        let reading = recognizer
            .read(
                &RgbImage::new(case.width, case.height, &pixels).unwrap(),
                case.kind,
            )
            .unwrap();
        assert_eq!(reading.text, case.text, "{}", case.file);
        assert!(
            (reading.confidence - case.confidence).abs() < 1e-3,
            "{}: confidence {} vs {}",
            case.file,
            reading.confidence,
            case.confidence
        );
    }
}

#[test]
fn decode_collapses_repeats_drops_blanks_and_keeps_the_weakest_frame() {
    let classes = ALPHABET.chars().count() + 1;
    let frame = |class: usize, p: f32| {
        let mut row = vec![(1.0 - p) / (classes - 1) as f32; classes];
        row[class] = p;
        row
    };
    // blank, 1, 1, blank, 1, 7, '.', 5, '~'
    let path = [
        (0, 0.99),
        (2, 0.9),
        (2, 0.95),
        (0, 0.6),
        (2, 0.9),
        (8, 0.97),
        (11, 0.9),
        (6, 0.9),
        (12, 0.8),
    ];
    let probabilities: Vec<f32> = path.iter().flat_map(|&(c, p)| frame(c, p)).collect();
    let reading = decode(&probabilities, classes, ALPHABET);
    assert_eq!(reading.text, "117.5~");
    assert!((reading.confidence - 0.6).abs() < 1e-6);
    assert!(reading.has_correction());
    assert_eq!(reading.value_text(), "117.5");
    assert!(decode(&frame(0, 0.9), classes, ALPHABET).is_blank());

    // A student number cannot contain '.' or '~': those frames fall back to
    // the best allowed class and their low probability shows in the confidence.
    let number = decode(&probabilities, classes, CellKind::StudentNumber.allowed());
    assert_eq!(number.text, "1175");
    assert!(number.confidence < 0.1);
}

#[test]
fn invalid_pixel_buffers_are_rejected() {
    assert!(RgbImage::new(2, 2, &[0; 11]).is_err());
    assert!(RgbImage::new(0, 2, &[]).is_err());
    let white = vec![255u8; 10 * 40 * 3];
    let tensor = preprocess(&RgbImage::new(40, 10, &white).unwrap());
    assert_eq!(tensor.width, 128);
    assert!(tensor.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}
