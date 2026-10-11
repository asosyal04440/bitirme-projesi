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

  String get subtitle => [
    info.courseCode,
    info.examKind,
    if (info.group != null) 'Grup ${info.group}',
    [info.academicYear, info.term].whereType<String>().join(' '),
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

class AppState extends ChangeNotifier {
  Storage? _storage;
  final List<ExamEntry> exams = [];
  int correctionCount = 0;

  Storage get storage => _storage!;

  /// Restores saved exams and sheets.
  Future<void> load() async {
    _storage = await Storage.open();
    exams.clear();
    for (final stored in await storage.exams()) {
      final entry = ExamEntry(
        stored: stored,
        info: await inspectExam(docx: stored.docx),
      );
      entry.sheets.addAll(await storage.sheets(stored.id));
      exams.add(entry);
    }
    correctionCount = await storage.correctionCount();
    notifyListeners();
  }

  /// Validates the exam by building its reader, then saves it.
  Future<ExamEntry> createExam({
    required Uint8List docx,
    required Map<String, String> maximums,
    required List<RosterStudent> roster,
  }) async {
    final stored = StoredExam(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      createdAt: DateTime.now(),
      docx: docx,
      maximums: maximums,
      roster: roster,
    );
    final entry = ExamEntry(
      stored: stored,
      info: await inspectExam(docx: docx),
    );
    await entry
        .reader(); // fails here, before saving, if the exam is incomplete
    await storage.insertExam(stored);
    exams.insert(0, entry);
    notifyListeners();
    return entry;
  }

  Future<void> deleteExam(ExamEntry exam) async {
    await storage.deleteExam(exam.id);
    exams.remove(exam);
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
      (await rootBundle.load('assets/digit-crnn-v1.onnx')).buffer.asUint8List(),
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
