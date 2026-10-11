//! The exam's Excel report. PÇ figures use only the scores the domain
//! accepted; a PÇ that is still incomplete on a sheet is left out of the class
//! totals and marked on that sheet's row.

use crate::api::reader::{ReportSheet, TOTAL_FIELD};
use exam_core::domain::{Exam, LearningOutcomeId, QuestionId, ScoreValue};
use exam_import::ImportedExam;
use rust_xlsxwriter::{Format, FormatAlign, FormatBorder, Workbook, Worksheet, XlsxError};

fn points(value: Option<&str>) -> Option<f64> {
    let value = value?.trim().replace(',', ".");
    ScoreValue::from_decimal(&value)
        .ok()
        .map(|v| v.micro_points() as f64 / 1_000_000.0)
}

fn value_of<'a>(sheet: &'a ReportSheet, field: &str) -> Option<&'a str> {
    sheet
        .values
        .iter()
        .find(|v| v.field == field)
        .and_then(|v| v.value.as_deref())
}

struct Formats {
    header: Format,
    percent: Format,
    number: Format,
    muted: Format,
    title: Format,
}

impl Formats {
    fn new() -> Self {
        let header = Format::new()
            .set_bold()
            .set_background_color("DCE8EC")
            .set_border_bottom(FormatBorder::Thin)
            .set_align(FormatAlign::Center);
        Self {
            header,
            percent: Format::new().set_num_format("0.0%"),
            number: Format::new().set_num_format("0.##"),
            muted: Format::new().set_font_color("888888"),
            title: Format::new().set_bold().set_font_size(14),
        }
    }
}

fn header_row(sheet: &mut Worksheet, titles: &[String], format: &Format) -> Result<(), XlsxError> {
    for (column, title) in titles.iter().enumerate() {
        sheet.write_string_with_format(0, column as u16, title, format)?;
    }
    sheet.set_freeze_panes(1, 0)?;
    Ok(())
}

pub(crate) fn workbook(
    imported: &ImportedExam,
    exam: &Exam,
    sheets: &[ReportSheet],
) -> anyhow::Result<Vec<u8>> {
    let f = Formats::new();
    let questions: Vec<(&str, &QuestionId, ScoreValue)> = imported
        .questions
        .iter()
        .filter_map(|q| Some((q.printed_id.as_str(), &q.id, q.maximum_score?)))
        .collect();
    let outcome_ids: Vec<&LearningOutcomeId> = exam.learning_outcomes.keys().collect();
    let to_f64 = |v: ScoreValue| v.micro_points() as f64 / 1_000_000.0;

    // Grades: one row per sheet.
    let mut grades = Worksheet::new();
    grades.set_name("Notlar")?;
    let mut titles = vec!["Öğrenci No".to_owned(), "Ad Soyad".to_owned()];
    titles.extend(questions.iter().map(|(id, _, max)| format!("{id} ({max})")));
    let total_max: f64 = questions.iter().map(|(_, _, m)| to_f64(*m)).sum();
    titles.push(format!("Toplam ({total_max})"));
    titles.push("Kâğıttaki Not".to_owned());
    titles.extend(outcome_ids.iter().map(|o| format!("{} başarı", o.as_str())));
    titles.push("Durum".to_owned());
    header_row(&mut grades, &titles, &f.header)?;
    for (index, sheet) in sheets.iter().enumerate() {
        let row = index as u32 + 1;
        let mut column = 0_u16;
        grades.write_string(row, column, sheet.student_number.as_deref().unwrap_or("?"))?;
        column += 1;
        grades.write_string(row, column, sheet.student_name.as_deref().unwrap_or(""))?;
        column += 1;
        let mut total = Some(0.0);
        for (id, _, _) in &questions {
            match points(value_of(sheet, id)) {
                Some(v) => {
                    total = total.map(|t| t + v);
                    grades.write_number_with_format(row, column, v, &f.number)?;
                }
                None => {
                    // Unconfirmed sheets only report accepted values.
                    let text = if sheet.confirmed { "boş" } else { "kontrol" };
                    grades.write_string_with_format(row, column, text, &f.muted)?;
                    total = None;
                }
            }
            column += 1;
        }
        // A partial sum would read as a grade; show it only when complete.
        match total {
            Some(total) => grades.write_number_with_format(row, column, total, &f.number)?,
            None => grades.write_string_with_format(row, column, "eksik", &f.muted)?,
        };
        column += 1;
        match points(value_of(sheet, TOTAL_FIELD)) {
            Some(v) => grades.write_number_with_format(row, column, v, &f.number)?,
            None => grades.write_string_with_format(row, column, "–", &f.muted)?,
        };
        column += 1;
        for outcome in &outcome_ids {
            match sheet.outcomes.iter().find(|o| o.id == outcome.as_str()) {
                Some(o) if o.complete => {
                    grades.write_number_with_format(row, column, o.percent / 100.0, &f.percent)?
                }
                _ => grades.write_string_with_format(row, column, "eksik", &f.muted)?,
            };
            column += 1;
        }
        grades.write_string(
            row,
            column,
            if sheet.confirmed {
                "Onaylandı"
            } else {
                "Kontrol bekliyor"
            },
        )?;
    }
    grades.set_column_width(0, 14)?;
    grades.set_column_width(1, 24)?;
    grades.set_column_range_width(2, titles.len() as u16 - 1, 12)?;

    // Class result per PÇ.
    let mut outcomes = Worksheet::new();
    outcomes.set_name("PÇ Analizi")?;
    header_row(
        &mut outcomes,
        &[
            "PÇ",
            "İlgili sorular",
            "Alınan puan",
            "Olası puan",
            "Başarı",
            "Kâğıt sayısı",
        ]
        .map(str::to_owned),
        &f.header,
    )?;
    for (index, outcome) in outcome_ids.iter().enumerate() {
        let row = index as u32 + 1;
        let related: Vec<String> = exam
            .questions
            .values()
            .filter(|q| {
                q.outcome_allocations
                    .iter()
                    .any(|a| &&a.outcome_id == outcome)
            })
            .filter_map(|q| {
                questions
                    .iter()
                    .find(|(_, id, _)| *id == &q.id)
                    .map(|(printed, _, _)| (*printed).to_owned())
            })
            .collect();
        let (mut earned, mut possible, mut count) = (0.0, 0.0, 0);
        for sheet in sheets {
            if let Some(o) = sheet
                .outcomes
                .iter()
                .find(|o| o.id == outcome.as_str() && o.complete)
            {
                earned += points(Some(&o.earned)).unwrap_or(0.0);
                possible += points(Some(&o.possible)).unwrap_or(0.0);
                count += 1;
            }
        }
        outcomes.write_string(row, 0, outcome.as_str())?;
        outcomes.write_string(row, 1, related.join(", "))?;
        outcomes.write_number_with_format(row, 2, earned, &f.number)?;
        outcomes.write_number_with_format(row, 3, possible, &f.number)?;
        if possible > 0.0 {
            outcomes.write_number_with_format(row, 4, earned / possible, &f.percent)?;
        } else {
            outcomes.write_string_with_format(row, 4, "–", &f.muted)?;
        }
        outcomes.write_number(row, 5, count as f64)?;
    }
    outcomes.set_column_width(1, 18)?;
    outcomes.set_column_range_width(2, 5, 13)?;

    // Class result per question.
    let mut by_question = Worksheet::new();
    by_question.set_name("Soru Analizi")?;
    header_row(
        &mut by_question,
        &[
            "Soru",
            "PÇ",
            "Tam puan",
            "Ortalama",
            "Ortalama başarı",
            "Okunan kâğıt",
        ]
        .map(str::to_owned),
        &f.header,
    )?;
    for (index, (printed, id, maximum)) in questions.iter().enumerate() {
        let row = index as u32 + 1;
        let values: Vec<f64> = sheets
            .iter()
            .filter_map(|s| points(value_of(s, printed)))
            .collect();
        let pcs: Vec<&str> = exam
            .questions
            .get(*id)
            .map(|q| {
                q.outcome_allocations
                    .iter()
                    .map(|a| a.outcome_id.as_str())
                    .collect()
            })
            .unwrap_or_default();
        by_question.write_string(row, 0, *printed)?;
        by_question.write_string(row, 1, pcs.join(", "))?;
        by_question.write_number_with_format(row, 2, to_f64(*maximum), &f.number)?;
        if values.is_empty() {
            by_question.write_string_with_format(row, 3, "–", &f.muted)?;
            by_question.write_string_with_format(row, 4, "–", &f.muted)?;
        } else {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            by_question.write_number_with_format(row, 3, mean, &f.number)?;
            by_question.write_number_with_format(row, 4, mean / to_f64(*maximum), &f.percent)?;
        }
        by_question.write_number(row, 5, values.len() as f64)?;
    }
    by_question.set_column_range_width(0, 5, 14)?;

    // Exam details.
    let mut details = Worksheet::new();
    details.set_name("Sınav Bilgisi")?;
    let m = &imported.metadata;
    details.write_string_with_format(0, 0, m.course.as_deref().unwrap_or("Sınav"), &f.title)?;
    let rows = [
        ("Ders kodu", m.course_code.clone()),
        ("Ders adı", m.course_name.clone()),
        ("Akademik yıl", m.academic_year.clone()),
        ("Dönem", m.term.clone()),
        ("Sınav", m.exam_kind.clone()),
        ("Grup", m.group.clone()),
        ("Tarih", m.date.clone()),
        ("Kâğıt sayısı", Some(sheets.len().to_string())),
        (
            "Onaylanan kâğıt",
            Some(sheets.iter().filter(|s| s.confirmed).count().to_string()),
        ),
    ];
    for (index, (label, value)) in rows.iter().enumerate() {
        let row = index as u32 + 2;
        details.write_string_with_format(row, 0, *label, &f.header)?;
        details.write_string(row, 1, value.as_deref().unwrap_or("–"))?;
    }
    let note_row = rows.len() as u32 + 3;
    details.write_string_with_format(
        note_row,
        0,
        "PÇ başarısı = Σ(soru puanı × PÇ ağırlığı) / Σ(tam puan × PÇ ağırlığı). \
         Puanı kesinleşmemiş PÇ'ler ilgili kâğıtta 'eksik' gösterilir ve sınıf toplamına katılmaz.",
        &f.muted,
    )?;
    details.set_column_width(0, 18)?;
    details.set_column_width(1, 36)?;

    let mut workbook = Workbook::new();
    workbook.push_worksheet(grades);
    workbook.push_worksheet(outcomes);
    workbook.push_worksheet(by_question);
    workbook.push_worksheet(details);
    Ok(workbook.save_to_buffer()?)
}
