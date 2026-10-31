import 'dart:convert';
import 'dart:typed_data';

import 'package:sqflite/sqflite.dart';

import 'src/rust/api/reader.dart';

/// A course the instructor teaches; its exams and sheets are kept under it.
class StoredCourse {
  StoredCourse({
    required this.id,
    required this.code,
    required this.name,
    required this.createdAt,
  });

  final String id;
  final String code;
  final String? name;
  final DateTime createdAt;
}

/// An exam as stored: the Word file and what the instructor added to it.
class StoredExam {
  StoredExam({
    required this.id,
    required this.courseId,
    required this.createdAt,
    required this.docx,
    required this.maximums,
    required this.roster,
    this.definition,
  });

  final String id;

  /// Null only for exams saved before courses existed; [AppState.load]
  /// assigns them to a course.
  final String? courseId;
  final DateTime createdAt;

  /// The exam's Word file; empty for an exam recognised on the sheet.
  final Uint8List docx;
  final Map<String, String> maximums;
  final List<RosterStudent> roster;

  /// The exam as recognised on the sheet (see `definition_json.dart`);
  /// null for an exam added from its Word file.
  final Map<String, dynamic>? definition;
}

/// A read sheet; `confirmed` once the instructor checked every value.
class SheetRecord {
  SheetRecord({
    required this.id,
    required this.createdAt,
    required this.confirmed,
    required this.result,
  });

  final String id;
  final DateTime createdAt;
  final bool confirmed;
  final SheetResult result;
}

/// A cell the instructor corrected: the crop, what the model read and the
/// right value. Kept on the device to retrain the recognizer later.
class Correction {
  Correction({
    required this.examId,
    required this.sheetId,
    required this.field,
    required this.recognized,
    required this.corrected,
    required this.confidence,
    required this.crop,
  });

  final String examId;
  final String sheetId;
  final String field;
  final String recognized;
  final String? corrected;
  final double confidence;
  final Uint8List crop;
}

class Storage {
  Storage._(this._db);

  final Database _db;

  static Future<Storage> open() async {
    final directory = await getDatabasesPath();
    final db = await openDatabase(
      '$directory/sinav_okuyucu.db',
      version: 3,
      onConfigure: (db) => db.execute('PRAGMA foreign_keys = ON'),
      onUpgrade: (db, from, _) async {
        if (from < 2) {
          await _createCourses(db);
          await db.execute(
            'ALTER TABLE exams ADD COLUMN course_id TEXT '
            'REFERENCES courses(id) ON DELETE CASCADE',
          );
        }
        if (from < 3) {
          await db.execute('ALTER TABLE exams ADD COLUMN definition TEXT');
        }
      },
      onCreate: (db, _) async {
        await _createCourses(db);
        await db.execute('''
          CREATE TABLE exams (
            id TEXT PRIMARY KEY,
            course_id TEXT REFERENCES courses(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            docx BLOB NOT NULL,
            maximums TEXT NOT NULL,
            roster TEXT NOT NULL,
            definition TEXT
          )''');
        await db.execute('''
          CREATE TABLE sheets (
            id TEXT PRIMARY KEY,
            exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            confirmed INTEGER NOT NULL,
            result TEXT NOT NULL
          )''');
        await db.execute('''
          CREATE TABLE crops (
            sheet_id TEXT NOT NULL REFERENCES sheets(id) ON DELETE CASCADE,
            field TEXT NOT NULL,
            png BLOB NOT NULL,
            PRIMARY KEY (sheet_id, field)
          )''');
        await db.execute('''
          CREATE TABLE corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            sheet_id TEXT NOT NULL,
            field TEXT NOT NULL,
            recognized TEXT NOT NULL,
            corrected TEXT,
            confidence REAL NOT NULL,
            png BLOB NOT NULL,
            created_at INTEGER NOT NULL
          )''');
      },
    );
    return Storage._(db);
  }

  static Future<void> _createCourses(Database db) => db.execute('''
    CREATE TABLE courses (
      id TEXT PRIMARY KEY,
      code TEXT NOT NULL UNIQUE,
      name TEXT,
      created_at INTEGER NOT NULL
    )''');

  Future<void> insertCourse(StoredCourse course) => _db.insert('courses', {
    'id': course.id,
    'code': course.code,
    'name': course.name,
    'created_at': course.createdAt.millisecondsSinceEpoch,
  });

  Future<void> renameCourse(String id, String? name) =>
      _db.update('courses', {'name': name}, where: 'id = ?', whereArgs: [id]);

  Future<List<StoredCourse>> courses() async {
    final rows = await _db.query('courses', orderBy: 'code');
    return [
      for (final row in rows)
        StoredCourse(
          id: row['id']! as String,
          code: row['code']! as String,
          name: row['name'] as String?,
          createdAt: DateTime.fromMillisecondsSinceEpoch(
            row['created_at']! as int,
          ),
        ),
    ];
  }

  Future<void> deleteCourse(String id) =>
      _db.delete('courses', where: 'id = ?', whereArgs: [id]);

  Future<void> moveExam(String examId, String courseId) => _db.update(
    'exams',
    {'course_id': courseId},
    where: 'id = ?',
    whereArgs: [examId],
  );

  Future<void> insertExam(StoredExam exam) => _db.insert('exams', {
    'id': exam.id,
    'course_id': exam.courseId,
    'created_at': exam.createdAt.millisecondsSinceEpoch,
    'docx': exam.docx,
    'maximums': jsonEncode(exam.maximums),
    'roster': jsonEncode([
      for (final s in exam.roster) {'number': s.number, 'name': s.name},
    ]),
    'definition': exam.definition == null ? null : jsonEncode(exam.definition),
  });

  Future<List<StoredExam>> exams() async {
    final rows = await _db.query('exams', orderBy: 'created_at DESC');
    return [
      for (final row in rows)
        StoredExam(
          id: row['id']! as String,
          courseId: row['course_id'] as String?,
          createdAt: DateTime.fromMillisecondsSinceEpoch(
            row['created_at']! as int,
          ),
          docx: row['docx']! as Uint8List,
          maximums: Map<String, String>.from(
            jsonDecode(row['maximums']! as String) as Map,
          ),
          roster: [
            for (final s in jsonDecode(row['roster']! as String) as List)
              RosterStudent(
                number: s['number'] as String,
                name: s['name'] as String,
              ),
          ],
          definition: row['definition'] == null
              ? null
              : jsonDecode(row['definition']! as String)
                    as Map<String, dynamic>,
        ),
    ];
  }

  /// Stores what the instructor changed in an exam recognised on the sheet
  /// (kind, PÇ, maximum scores).
  Future<void> updateDefinition(String id, Map<String, dynamic> definition) =>
      _db.update(
        'exams',
        {'definition': jsonEncode(definition)},
        where: 'id = ?',
        whereArgs: [id],
      );

  Future<void> deleteExam(String id) =>
      _db.delete('exams', where: 'id = ?', whereArgs: [id]);

  Future<void> saveSheet(String examId, SheetRecord record) =>
      _db.transaction((txn) async {
        await txn.insert('sheets', {
          'id': record.id,
          'exam_id': examId,
          'created_at': record.createdAt.millisecondsSinceEpoch,
          'confirmed': record.confirmed ? 1 : 0,
          'result': jsonEncode(sheetToJson(record.result)),
        }, conflictAlgorithm: ConflictAlgorithm.replace);
        for (final cell in record.result.cells) {
          if (cell.cropPng.isEmpty) continue;
          await txn.insert('crops', {
            'sheet_id': record.id,
            'field': cell.field,
            'png': cell.cropPng,
          }, conflictAlgorithm: ConflictAlgorithm.replace);
        }
      });

  Future<List<SheetRecord>> sheets(String examId) async {
    final rows = await _db.query(
      'sheets',
      where: 'exam_id = ?',
      whereArgs: [examId],
      orderBy: 'created_at DESC',
    );
    final records = <SheetRecord>[];
    for (final row in rows) {
      final id = row['id']! as String;
      final crops = {
        for (final c in await _db.query(
          'crops',
          where: 'sheet_id = ?',
          whereArgs: [id],
        ))
          c['field']! as String: c['png']! as Uint8List,
      };
      records.add(
        SheetRecord(
          id: id,
          createdAt: DateTime.fromMillisecondsSinceEpoch(
            row['created_at']! as int,
          ),
          confirmed: row['confirmed'] == 1,
          result: sheetFromJson(
            jsonDecode(row['result']! as String) as Map<String, dynamic>,
            crops,
          ),
        ),
      );
    }
    return records;
  }

  Future<void> deleteSheet(String id) =>
      _db.delete('sheets', where: 'id = ?', whereArgs: [id]);

  Future<void> addCorrections(List<Correction> corrections) =>
      _db.transaction((txn) async {
        final now = DateTime.now().millisecondsSinceEpoch;
        for (final c in corrections) {
          await txn.insert('corrections', {
            'exam_id': c.examId,
            'sheet_id': c.sheetId,
            'field': c.field,
            'recognized': c.recognized,
            'corrected': c.corrected,
            'confidence': c.confidence,
            'png': c.crop,
            'created_at': now,
          });
        }
      });

  Future<int> correctionCount() async =>
      Sqflite.firstIntValue(
        await _db.rawQuery('SELECT COUNT(*) FROM corrections'),
      ) ??
      0;
}

Map<String, dynamic> sheetToJson(SheetResult r) => {
  'sheetId': r.sheetId,
  'pageFound': r.pageFound,
  'studentNumber': r.studentNumber,
  'studentName': r.studentName,
  'requiresReview': r.requiresReview,
  'cells': [
    for (final c in r.cells)
      {
        'field': c.field,
        'text': c.text,
        'value': c.value,
        'confidence': c.confidence,
        'accepted': c.accepted,
        'region': c.region.toList(),
      },
  ],
  'issues': [
    for (final i in r.issues)
      {'reason': i.reason, 'field': i.field, 'detail': i.detail},
  ],
  'outcomes': [
    for (final o in r.outcomes)
      {
        'id': o.id,
        'earned': o.earned,
        'possible': o.possible,
        'percent': o.percent,
        'complete': o.complete,
        'maximumKnown': o.maximumKnown,
      },
  ],
};

SheetResult sheetFromJson(
  Map<String, dynamic> json,
  Map<String, Uint8List> crops,
) => SheetResult(
  sheetId: json['sheetId'] as String,
  pageFound: json['pageFound'] as bool,
  studentNumber: json['studentNumber'] as String?,
  studentName: json['studentName'] as String?,
  requiresReview: json['requiresReview'] as bool,
  cells: [
    for (final c in json['cells'] as List)
      CellResult(
        field: c['field'] as String,
        text: c['text'] as String,
        value: c['value'] as String?,
        confidence: (c['confidence'] as num).toDouble(),
        accepted: c['accepted'] as bool? ?? false,
        region: Float32List.fromList([
          for (final v in c['region'] as List) (v as num).toDouble(),
        ]),
        cropPng: crops[c['field']] ?? Uint8List(0),
      ),
  ],
  issues: [
    for (final i in json['issues'] as List)
      IssueResult(
        reason: i['reason'] as String,
        field: i['field'] as String,
        detail: i['detail'] as String?,
      ),
  ],
  outcomes: [
    for (final o in json['outcomes'] as List)
      OutcomeResult(
        id: o['id'] as String,
        earned: o['earned'] as String,
        possible: o['possible'] as String,
        percent: (o['percent'] as num).toDouble(),
        complete: o['complete'] as bool,
        // Sheets saved before maximum scores could be left out had them.
        maximumKnown: o['maximumKnown'] as bool? ?? true,
      ),
  ],
);
