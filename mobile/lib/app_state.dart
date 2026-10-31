import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'definition_json.dart';
import 'src/rust/api/definition.dart';
import 'src/rust/api/exam.dart';
import 'src/rust/api/reader.dart';
import 'src/rust/api/scanner.dart';
import 'storage.dart';

/// One exam the instructor added, with its sheets.
class ExamEntry {
  ExamEntry({required this.stored, required this.info});

  final StoredExam stored;
  final ExamInfo info;
  final List<SheetRecord> sheets = [];
  Future<ExamReader>? _reader;

  String get id => stored.id;
  int get rosterSize => stored.roster.length;
  String get title => info.courseName ?? info.course ?? 'Sınav';

  /// "Final · Grup A" — how the exam is told apart inside its course.
  String get label => [
    info.examKind ?? 'Sınav',
    if (info.group != null) 'Grup ${info.group}',
  ].join(' · ');

  /// "2025-2026 Bahar · 10/04/2026 - 11.00"
  String get subtitle => [
    [info.academicYear, info.term].whereType<String>().join(' '),
    info.date,
  ].whereType<String>().where((s) => s.isNotEmpty).join(' · ');

  /// Maximum score per printed question id, from the file or the instructor.
  Map<String, String> get maximums => {
    for (final q in info.questions)
      if ((q.maximumScore ?? stored.maximums[q.id]) != null)
        q.id: (q.maximumScore ?? stored.maximums[q.id])!,
  };

  /// Some question has no maximum score yet (it is not printed on the
  /// sheet): its PÇ are shown in points until the instructor enters it.
  bool get maximumsMissing => info.questions.any((q) => q.maximumScore == null);

  /// Exam details the instructor may still fill in after scanning started.
  bool get detailsMissing => maximumsMissing || info.examKind == null;

  /// The reader loads the recognizer model, so it is built on first use.
  Future<ExamReader> reader() => _reader ??= _createReader(stored);

  static Future<ExamReader> _createReader(StoredExam stored) async {
    final assets = await BundledAssets.load();
    final maximumScores = [
      for (final e in stored.maximums.entries)
        MaximumScore(question: e.key, value: e.value),
    ];
    if (stored.definition != null) {
      return ExamReader.fromDefinition(
        templateJson: assets.templateJson,
        templatePng: assets.templatePng,
        modelOnnx: assets.model,
        definition: definitionFromJson(stored.definition!),
        maximumScores: maximumScores,
        roster: stored.roster,
      );
    }
    return ExamReader.newInstance(
      templateJson: assets.templateJson,
      templatePng: assets.templatePng,
      modelOnnx: assets.model,
      examDocx: stored.docx,
      maximumScores: [
        for (final e in stored.maximums.entries)
          MaximumScore(question: e.key, value: e.value),
      ],
      roster: stored.roster,
    );
  }
}

/// A course with its exams, newest first.
class CourseEntry {
  CourseEntry(this.stored);

  StoredCourse stored;
  final List<ExamEntry> exams = [];

  String get id => stored.id;
  String get code => stored.code;

  /// The name the instructor typed, or the one read from an exam's Word file.
  String? get name =>
      stored.name ??
      exams.map((e) => e.info.courseName).whereType<String>().firstOrNull;

  int get sheetCount => exams.fold(0, (n, e) => n + e.sheets.length);
  int get pendingCount =>
      exams.fold(0, (n, e) => n + e.sheets.where((s) => !s.confirmed).length);
}

/// Course codes are compared without spaces and case ("bilm 374" = "BİLM374").
String normalizeCourseCode(String code) {
  final compact = code.replaceAll(RegExp(r'\s+'), '');
  return compact.replaceAll('i', 'İ').replaceAll('ı', 'I').toUpperCase();
}

/// Why an exam's Word file cannot be added to [course], or null if it can.
String? courseMismatch(CourseEntry course, ExamInfo info) {
  final code = info.courseCode;
  if (code == null || normalizeCourseCode(code) == course.code) return null;
  return 'Bu Word dosyası $code dersine ait; seçili ders ${course.code}.';
}

class AppState extends ChangeNotifier {
  Storage? _storage;
  final List<CourseEntry> courses = [];
  int correctionCount = 0;

  Storage get storage => _storage!;

  /// Every exam of every course.
  List<ExamEntry> get exams => [for (final c in courses) ...c.exams];

  /// Restores saved courses, exams and sheets.
  Future<void> load() async {
    _storage = await Storage.open();
    courses
      ..clear()
      ..addAll([for (final c in await storage.courses()) CourseEntry(c)]);
    for (final stored in await storage.exams()) {
      final info = await examInfo(stored);
      final entry = ExamEntry(stored: stored, info: info);
      entry.sheets.addAll(await storage.sheets(stored.id));
      var course = courses.where((c) => c.id == stored.courseId).firstOrNull;
      if (course == null) {
        // Saved before courses existed: file it under its Word course code.
        course = await _courseFor(info.courseCode ?? 'DİĞER', info.courseName);
        await storage.moveExam(stored.id, course.id);
      }
      course.exams.add(entry);
    }
    correctionCount = await storage.correctionCount();
    notifyListeners();
  }

  /// What the app shows for a stored exam, from its Word file or from the
  /// definition recognised on the sheet.
  static Future<ExamInfo> examInfo(StoredExam stored) async =>
      stored.definition != null
      ? definitionInfo(definition: definitionFromJson(stored.definition!))
      : await inspectExam(docx: stored.docx);

  /// The saved exam a recognised sheet belongs to: same course code, kind,
  /// group and term (fields that could not be read are not compared).
  ExamEntry? examFor(ExamDefinition definition) {
    final code = definition.courseCode;
    if (code == null) return null;
    final course = courses
        .where((c) => c.code == normalizeCourseCode(code))
        .firstOrNull;
    bool same(String? a, String? b) => a == null || b == null || a == b;
    return course?.exams
        .where(
          (e) =>
              same(e.info.examKind, definition.examKind) &&
              same(e.info.group, definition.group) &&
              same(e.info.academicYear, definition.academicYear) &&
              same(e.info.term, definition.term),
        )
        .firstOrNull;
  }

  /// Saves an exam recognised on its first sheet, under its course (created
  /// from the course code when new). Scanning can start right away; what the
  /// sheet does not say (maximum scores) is filled in later.
  Future<ExamEntry> createDetectedExam(ExamDefinition definition) async {
    final course = await _courseFor(
      definition.courseCode ?? 'DİĞER',
      definition.courseName,
    );
    final stored = StoredExam(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      courseId: course.id,
      createdAt: DateTime.now(),
      docx: Uint8List(0),
      maximums: const {},
      roster: const [],
      definition: definitionToJson(definition),
    );
    final entry = ExamEntry(stored: stored, info: await examInfo(stored));
    await entry
        .reader(); // fails here, before saving, if the exam is incomplete
    await storage.insertExam(stored);
    course.exams.insert(0, entry);
    notifyListeners();
    return entry;
  }

  /// Applies what the instructor changed in a recognised exam (course code,
  /// kind, group, PÇ, maximum scores) and works out the PÇ results of its
  /// sheets again. Returns the exam as it is now.
  Future<ExamEntry> updateDetectedExam(
    ExamEntry exam,
    ExamDefinition definition,
  ) async {
    final old = exam.stored;
    final from = courses.firstWhere((c) => c.exams.contains(exam));
    final to = await _courseFor(
      definition.courseCode ?? from.code,
      definition.courseName,
    );
    final stored = StoredExam(
      id: old.id,
      courseId: to.id,
      createdAt: old.createdAt,
      docx: old.docx,
      maximums: old.maximums,
      roster: old.roster,
      definition: definitionToJson(definition),
    );
    final updated = ExamEntry(stored: stored, info: await examInfo(stored));
    final reader = await updated.reader();
    for (final record in exam.sheets) {
      final redone = await reader.confirmSheet(
        sheetId: record.id,
        values: [
          for (final c in record.result.cells)
            if (record.confirmed || c.accepted)
              FieldValue(field: c.field, value: c.value, region: c.region),
        ],
      );
      final r = record.result;
      // Only the PÇ results change; what was read stays as it was.
      final result = SheetResult(
        sheetId: r.sheetId,
        pageFound: r.pageFound,
        studentNumber: r.studentNumber,
        studentName: r.studentName,
        cells: r.cells,
        issues: r.issues,
        outcomes: redone.outcomes,
        requiresReview: r.requiresReview,
      );
      final again = SheetRecord(
        id: record.id,
        createdAt: record.createdAt,
        confirmed: record.confirmed,
        result: result,
      );
      await storage.saveSheet(stored.id, again);
      updated.sheets.add(again);
    }
    await storage.updateDefinition(stored.id, stored.definition!);
    if (to != from) await storage.moveExam(stored.id, to.id);
    final index = from.exams.indexOf(exam);
    from.exams.removeAt(index);
    to.exams.insert(to == from ? index : 0, updated);
    notifyListeners();
    return updated;
  }

  SheetScanner? _scanner;

  /// Recognises the exam on a sheet shown to the camera; built on first use
  /// (it loads the printed-text model).
  Future<SheetScanner> scanner() async {
    if (_scanner != null) return _scanner!;
    final assets = await BundledAssets.load();
    return _scanner = await SheetScanner.newInstance(
      templateJson: assets.templateJson,
      templatePng: assets.templatePng,
      ocrModel: (await rootBundle.load('assets/ppocr-v4-rec.onnx')).buffer
          .asUint8List(),
      ocrKeys: await rootBundle.loadString('assets/ppocr-v4-rec.keys.txt'),
    );
  }

  Future<CourseEntry> _courseFor(String code, String? name) async {
    final normalized = normalizeCourseCode(code);
    return courses.where((c) => c.code == normalized).firstOrNull ??
        await createCourse(code: normalized, name: name);
  }

  /// Adds a course; the code must be new.
  Future<CourseEntry> createCourse({required String code, String? name}) async {
    final normalized = normalizeCourseCode(code);
    if (normalized.isEmpty) throw ArgumentError('Ders kodu boş olamaz.');
    if (courses.any((c) => c.code == normalized)) {
      throw StateError('$normalized dersi zaten var.');
    }
    final trimmed = name?.trim();
    final course = CourseEntry(
      StoredCourse(
        id: DateTime.now().microsecondsSinceEpoch.toString(),
        code: normalized,
        name: trimmed == null || trimmed.isEmpty ? null : trimmed,
        createdAt: DateTime.now(),
      ),
    );
    await storage.insertCourse(course.stored);
    courses
      ..add(course)
      ..sort((a, b) => a.code.compareTo(b.code));
    notifyListeners();
    return course;
  }

  /// Deletes the course with all its exams and sheets.
  Future<void> deleteCourse(CourseEntry course) async {
    await storage.deleteCourse(course.id);
    courses.remove(course);
    notifyListeners();
  }

  /// Validates the exam by building its reader, then saves it in [course].
  Future<ExamEntry> createExam({
    required CourseEntry course,
    required Uint8List docx,
    required Map<String, String> maximums,
    required List<RosterStudent> roster,
  }) async {
    final info = await inspectExam(docx: docx);
    final mismatch = courseMismatch(course, info);
    if (mismatch != null) throw StateError(mismatch);
    final stored = StoredExam(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      courseId: course.id,
      createdAt: DateTime.now(),
      docx: docx,
      maximums: maximums,
      roster: roster,
    );
    final entry = ExamEntry(stored: stored, info: info);
    await entry
        .reader(); // fails here, before saving, if the exam is incomplete
    await storage.insertExam(stored);
    course.exams.insert(0, entry);
    notifyListeners();
    return entry;
  }

  Future<void> deleteExam(ExamEntry exam) async {
    await storage.deleteExam(exam.id);
    for (final course in courses) {
      course.exams.remove(exam);
    }
    notifyListeners();
  }

  Future<void> saveSheet(ExamEntry exam, SheetRecord record) async {
    await storage.saveSheet(exam.id, record);
    exam.sheets.removeWhere((s) => s.id == record.id);
    exam.sheets.insert(0, record);
    notifyListeners();
  }

  Future<void> deleteSheet(ExamEntry exam, SheetRecord record) async {
    await storage.deleteSheet(record.id);
    exam.sheets.removeWhere((s) => s.id == record.id);
    notifyListeners();
  }

  Future<void> addCorrections(List<Correction> corrections) async {
    if (corrections.isEmpty) return;
    await storage.addCorrections(corrections);
    correctionCount += corrections.length;
    notifyListeners();
  }
}

final appState = AppState();

/// Sheet template and recognizer model shipped with the app.
class BundledAssets {
  BundledAssets._(this.templateJson, this.templatePng, this.model);

  final String templateJson;
  final Uint8List templatePng;
  final Uint8List model;

  static BundledAssets? _cache;

  static Future<BundledAssets> load() async {
    return _cache ??= BundledAssets._(
      await rootBundle.loadString('assets/inonu-final-a.json'),
      (await rootBundle.load('assets/inonu-final-a.png')).buffer.asUint8List(),
      (await rootBundle.load('assets/digit-crnn-v4.onnx')).buffer.asUint8List(),
    );
  }
}

/// Parses a roster file: one student per line, `number;name` or
/// `number,name`; a header line without digits is skipped.
List<RosterStudent> parseRoster(String text) {
  final students = <RosterStudent>[];
  for (final raw in const LineSplitter().convert(text)) {
    final line = raw.trim();
    if (line.isEmpty) continue;
    final parts = line.split(RegExp(r'[;,\t]'));
    if (parts.length < 2) continue;
    final number = parts.first.trim();
    final name = parts.sublist(1).join(' ').trim();
    if (!RegExp(r'^\d+$').hasMatch(number) || name.isEmpty) continue;
    students.add(RosterStudent(number: number, name: name));
  }
  return students;
}
