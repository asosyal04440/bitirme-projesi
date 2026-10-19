import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'src/rust/api/exam.dart';
import 'src/rust/api/reader.dart';
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

  /// The reader loads the recognizer model, so it is built on first use.
  Future<ExamReader> reader() => _reader ??= _createReader(stored);

  static Future<ExamReader> _createReader(StoredExam stored) async {
    final assets = await BundledAssets.load();
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
      final info = await inspectExam(docx: stored.docx);
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
      (await rootBundle.load('assets/digit-crnn-v3.onnx')).buffer.asUint8List(),
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
