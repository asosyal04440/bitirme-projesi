//! A minimal reader for the parts of a Word (.docx) file an exam needs:
//! paragraph texts and tables (with column widths) from the body and the
//! headers, plus the page width and left margin. Formatting is ignored.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::fmt;
use std::io::{Cursor, Read};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportError {
    Archive(String),
    Xml(String),
    MissingPart(&'static str),
    NoQuestionTable,
    Incomplete(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive(reason) => write!(f, "not a readable .docx file: {reason}"),
            Self::Xml(reason) => write!(f, "malformed document XML: {reason}"),
            Self::MissingPart(part) => write!(f, "the document has no {part}"),
            Self::NoQuestionTable => write!(f, "no 'Soru-PÇ İlişkisi' table was found"),
            Self::Incomplete(reason) => write!(f, "exam definition is incomplete: {reason}"),
        }
    }
}

impl std::error::Error for ImportError {}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cell {
    pub paragraphs: Vec<String>,
    /// Preferred width in twentieths of a point, when given.
    pub width_twips: Option<u32>,
    pub grid_span: u32,
}

impl Cell {
    pub fn text(&self) -> String {
        self.paragraphs.join(" ").trim().to_owned()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    pub grid_twips: Vec<u32>,
    pub indent_twips: i32,
    pub rows: Vec<Vec<Cell>>,
}

/// The body or one header: every paragraph in reading order (including those
/// inside table cells) and every table (including nested ones).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Part {
    pub paragraphs: Vec<String>,
    pub tables: Vec<Table>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub body: Part,
    pub headers: Vec<Part>,
    pub page_width_twips: u32,
    pub left_margin_twips: u32,
}

pub fn read_docx(bytes: &[u8]) -> Result<Document, ImportError> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| ImportError::Archive(e.to_string()))?;
    let mut read = |name: &str| -> Result<Option<String>, ImportError> {
        let mut file = match archive.by_name(name) {
            Ok(file) => file,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(e) => return Err(ImportError::Archive(e.to_string())),
        };
        let mut text = String::new();
        file.read_to_string(&mut text)
            .map_err(|e| ImportError::Archive(e.to_string()))?;
        Ok(Some(text))
    };
    let body_xml =
        read("word/document.xml")?.ok_or(ImportError::MissingPart("word/document.xml"))?;
    let (body, page) = parse_part(&body_xml)?;
    let mut headers = Vec::new();
    for index in 1..=6 {
        if let Some(xml) = read(&format!("word/header{index}.xml"))? {
            headers.push(parse_part(&xml)?.0);
        }
    }
    Ok(Document {
        body,
        headers,
        page_width_twips: page.0.unwrap_or(11_906), // A4 when unspecified
        left_margin_twips: page.1.unwrap_or(1_417),
    })
}

fn attribute(element: &BytesStart<'_>, name: &str) -> Option<String> {
    element
        .try_get_attribute(name)
        .ok()
        .flatten()
        .map(|a| a.value.into_owned())
}

fn twips(element: &BytesStart<'_>, name: &str) -> Option<u32> {
    attribute(element, name)?.parse().ok()
}

#[derive(Default)]
struct TableBuilder {
    table: Table,
    row: Option<Vec<Cell>>,
    cell: Option<Cell>,
}

/// (page width, left margin) in twips, when a part has section properties.
type PageGeometry = (Option<u32>, Option<u32>);

/// Parses one part and its page geometry.
fn parse_part(xml: &str) -> Result<(Part, PageGeometry), ImportError> {
    let mut reader = Reader::from_str(xml);
    let mut part = Part::default();
    let mut tables: Vec<TableBuilder> = Vec::new();
    let mut paragraph: Option<String> = None;
    let mut in_text = false;
    let mut page = (None, None);
    loop {
        let event = reader
            .read_event()
            .map_err(|e| ImportError::Xml(e.to_string()))?;
        match event {
            Event::Start(element) => match element.name().as_ref() {
                "w:tbl" => tables.push(TableBuilder::default()),
                "w:tr" => {
                    if let Some(t) = tables.last_mut() {
                        t.row = Some(Vec::new());
                    }
                }
                "w:tc" => {
                    if let Some(t) = tables.last_mut() {
                        t.cell = Some(Cell {
                            grid_span: 1,
                            ..Cell::default()
                        });
                    }
                }
                "w:p" => paragraph = Some(String::new()),
                "w:t" => in_text = true,
                _ => {}
            },
            Event::Empty(element) => {
                let name = element.name();
                match name.as_ref() {
                    "w:gridCol" => {
                        if let (Some(t), Some(w)) = (tables.last_mut(), twips(&element, "w:w")) {
                            t.table.grid_twips.push(w);
                        }
                    }
                    "w:tblInd" => {
                        if let Some(t) = tables.last_mut() {
                            t.table.indent_twips = attribute(&element, "w:w")
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(0);
                        }
                    }
                    "w:tcW" => {
                        if let Some(cell) = tables.last_mut().and_then(|t| t.cell.as_mut()) {
                            cell.width_twips = twips(&element, "w:w");
                        }
                    }
                    "w:gridSpan" => {
                        if let Some(cell) = tables.last_mut().and_then(|t| t.cell.as_mut()) {
                            cell.grid_span = twips(&element, "w:val").unwrap_or(1).max(1);
                        }
                    }
                    "w:tab" | "w:br" | "w:cr" => {
                        if let Some(p) = paragraph.as_mut() {
                            p.push(' ');
                        }
                    }
                    "w:pgSz" => page.0 = twips(&element, "w:w"),
                    "w:pgMar" => page.1 = twips(&element, "w:left"),
                    _ => {}
                }
            }
            Event::Text(text) if in_text => {
                if let Some(p) = paragraph.as_mut() {
                    p.push_str(&text.into_inner());
                }
            }
            Event::GeneralRef(reference) if in_text => {
                if let Some(p) = paragraph.as_mut() {
                    let resolved = match reference.resolve_char_ref() {
                        Ok(Some(c)) => Some(c),
                        _ => match reference.into_inner().as_ref() {
                            "amp" => Some('&'),
                            "lt" => Some('<'),
                            "gt" => Some('>'),
                            "quot" => Some('"'),
                            "apos" => Some('\''),
                            _ => None,
                        },
                    };
                    p.extend(resolved);
                }
            }
            Event::End(element) => match element.name().as_ref() {
                "w:t" => in_text = false,
                "w:p" => {
                    if let Some(text) = paragraph.take() {
                        if let Some(cell) = tables.last_mut().and_then(|t| t.cell.as_mut()) {
                            cell.paragraphs.push(text.clone());
                        }
                        part.paragraphs.push(text);
                    }
                }
                "w:tc" => {
                    if let Some(t) = tables.last_mut() {
                        if let (Some(row), Some(cell)) = (t.row.as_mut(), t.cell.take()) {
                            row.push(cell);
                        }
                    }
                }
                "w:tr" => {
                    if let Some(t) = tables.last_mut() {
                        if let Some(row) = t.row.take() {
                            t.table.rows.push(row);
                        }
                    }
                }
                "w:tbl" => {
                    if let Some(builder) = tables.pop() {
                        part.tables.push(builder.table);
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }
    Ok((part, page))
}
