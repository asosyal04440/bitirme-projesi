//! PaddleOCR printed-text recognition through `tract`, compared with what the
//! printed sheet says.

use exam_infrastructure::printed::PrintedTextReader;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reader() -> PrintedTextReader {
    PrintedTextReader::from_bytes(
        std::fs::read(root().join("models/ppocr-v4-rec.onnx")).unwrap(),
        &std::fs::read_to_string(root().join("models/ppocr-v4-rec.keys.txt")).unwrap(),
    )
    .unwrap()
}

/// Template-pixel box (1700 x 2200 blank sheet) cut out with a small inset.
fn crop(x0: u32, y0: u32, x1: u32, y1: u32) -> image::RgbImage {
    let blank = image::open(root().join("tools/synthetic/templates/inonu-final-a.png"))
        .unwrap()
        .to_rgb8();
    image::imageops::crop_imm(&blank, x0 + 3, y0 + 2, x1 - x0 - 6, y1 - y0 - 4).to_image()
}

#[test]
fn printed_labels_of_the_sheet_are_read() {
    let reader = reader();
    let course = reader.read(&crop(335, 299, 822, 334)).unwrap();
    assert_eq!(
        course.text.replace(' ', "").to_uppercase(),
        "BILM374-2022-YAPAYZEKA",
        "{course:?}"
    );
    let date = reader.read(&crop(335, 186, 605, 240)).unwrap();
    assert_eq!(date.text.replace(' ', ""), "10/04/2026-11.00");
    // The second question's Soru-PÇ cell, "(S2)-PÇ1" (no Ç in the model's charset).
    let mapping = reader.read(&crop(440, 542, 596, 586)).unwrap();
    assert_eq!(mapping.text.replace(' ', ""), "(S2)-PC1", "{mapping:?}");
}
