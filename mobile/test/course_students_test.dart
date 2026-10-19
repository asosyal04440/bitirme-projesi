import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sinav_okuyucu/app_state.dart';
import 'package:sinav_okuyucu/course_students.dart';
import 'package:sinav_okuyucu/pages/students_tab.dart';
import 'package:sinav_okuyucu/src/rust/api/exam.dart';
import 'package:sinav_okuyucu/src/rust/api/reader.dart';
import 'package:sinav_okuyucu/storage.dart';

ExamEntry exam(String id, String kind) => ExamEntry(
  stored: StoredExam(
    id: id,
    courseId: 'c',
    createdAt: DateTime(2026),
    docx: Uint8List(0),
    maximums: const {},
    roster: const [],
  ),
  info: ExamInfo(
    courseCode: 'BİLM374',
    examKind: kind,
    questions: const [],
    warnings: const [],
    ready: true,
  ),
);

/// A sheet whose number cell holds [number]; outcomes as (id, earned, possible).
SheetRecord sheet(
  String id,
  String? number,
  List<(String, String, String)> outcomes, {
  bool confirmed = true,
  bool complete = true,
  int minute = 0,
}) => SheetRecord(
  id: id,
  createdAt: DateTime(2026, 1, 1, 10, minute),
  confirmed: confirmed,
  result: SheetResult(
    sheetId: id,
    pageFound: true,
    cells: [
      CellResult(
        field: 'Numara',
        text: number ?? '',
        value: number,
        confidence: 1,
        accepted: true,
        region: Float32List(0),
        cropPng: Uint8List(0),
      ),
    ],
    issues: const [],
    outcomes: [
      for (final (o, earned, possible) in outcomes)
        OutcomeResult(
          id: o,
          earned: earned,
          possible: possible,
          percent: double.parse(earned) / double.parse(possible) * 100,
          complete: complete,
        ),
    ],
    requiresReview: false,
  ),
);

void main() {
  late CourseEntry course;
  late ExamEntry vize, finalExam, but;

  setUp(() {
    course = CourseEntry(
      StoredCourse(
        id: 'c',
        code: 'BİLM374',
        name: null,
        createdAt: DateTime(2026),
      ),
    );
    vize = exam('v', 'Vize');
    finalExam = exam('f', 'Final');
    but = exam('b', 'Bütünleme');
    course.exams.addAll([finalExam, but, vize]);
  });

  test('sheets are grouped by student number across exams', () {
    vize.sheets.add(sheet('1', '111', [('PÇ1', '10', '20')]));
    finalExam.sheets.add(sheet('2', '111', [('PÇ1', '30', '40')]));
    finalExam.sheets.add(sheet('3', '222', [('PÇ1', '40', '40')]));
    finalExam.sheets.add(sheet('4', null, [('PÇ1', '1', '40')]));

    final (:students, :unidentified) = courseStudents(course);
    expect(students.map((s) => s.number), ['111', '222']);
    expect(students.first.exams.map((e) => e.kind), ['Vize', 'Final']);
    expect(unidentified.single.sheet.id, '4');

    // Vize %50, Final %75 → 0.4 × 50 + 0.6 × 75
    final total = students.first.totals.single;
    expect(total.percent, closeTo(65, 1e-9));
    expect(total.complete, isTrue);
  });

  test('bütünleme replaces final in the course PÇ', () {
    vize.sheets.add(sheet('1', '111', [('PÇ1', '10', '20')]));
    finalExam.sheets.add(sheet('2', '111', [('PÇ1', '5', '40')]));
    but.sheets.add(sheet('3', '111', [('PÇ1', '35', '40')]));

    final student = courseStudents(course).students.single;
    expect(student.exams.map((e) => e.kind), ['Vize', 'Final', 'Bütünleme']);
    expect(student.counted.map((e) => e.kind), ['Vize', 'Bütünleme']);
    // Vize %50, Bütünleme %87.5 → 0.4 × 50 + 0.6 × 87.5
    expect(student.totals.single.percent, closeTo(72.5, 1e-9));
  });

  test('a PÇ asked only in the final takes the final percent', () {
    vize.sheets.add(sheet('1', '111', [('PÇ1', '10', '20')]));
    finalExam.sheets.add(
      sheet('2', '111', [('PÇ1', '30', '40'), ('PÇ2', '15', '20')]),
    );
    final totals = courseStudents(course).students.single.totals;
    expect(totals.map((t) => t.id), ['PÇ1', 'PÇ2']);
    expect(totals[1].percent, closeTo(75, 1e-9));
    expect(totals[1].complete, isTrue);
  });

  test('the course PÇ waits for both the vize and the final', () {
    finalExam.sheets.add(sheet('1', '111', [('PÇ1', '30', '40')]));
    expect(
      courseStudents(course).students.single.totals.single.complete,
      false,
    );
  });

  test('a re-scanned sheet replaces the older one', () {
    finalExam.sheets.add(sheet('old', '111', [('PÇ1', '1', '40')]));
    finalExam.sheets.add(sheet('new', '111', [('PÇ1', '39', '40')], minute: 5));
    final student = courseStudents(course).students.single;
    expect(student.exams.single.sheet.id, 'new');
  });

  test('a PÇ with a score that is not final is incomplete', () {
    vize.sheets.add(sheet('0', '111', [('PÇ1', '10', '20')]));
    finalExam.sheets.add(
      sheet('1', '111', [('PÇ1', '10', '40')], complete: false),
    );
    expect(
      courseStudents(course).students.single.totals.single.complete,
      false,
    );
  });

  test('PÇ ids are sorted by number', () {
    expect((['PÇ10', 'PÇ2', 'PÇ1']..sort(compareOutcomeIds)), [
      'PÇ1',
      'PÇ2',
      'PÇ10',
    ]);
  });
}
