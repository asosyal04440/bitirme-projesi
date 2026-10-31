import 'app_state.dart';
import 'src/rust/api/reader.dart';
import 'storage.dart';

/// Order in which a course's exams are shown for a student.
const examKindOrder = ['Vize', 'Mazeret', 'Final', 'Bütünleme'];

/// One student's sheet in one exam of the course.
class StudentExam {
  StudentExam({required this.exam, required this.sheet});

  final ExamEntry exam;
  final SheetRecord sheet;

  String get kind => exam.info.examKind ?? 'Sınav';
  bool get confirmed => sheet.confirmed;
  List<OutcomeResult> get outcomes => sheet.result.outcomes;
}

/// Weights of the exams in the course PÇ (department rule). Mazeret counts
/// as the Vize, Bütünleme as the Final.
const examWeights = {'Vize': 0.4, 'Final': 0.6};

/// A student's course-level achievement of one PÇ.
class OutcomeTotal {
  OutcomeTotal(this.id);

  final String id;
  double _weighted = 0;
  double _weights = 0;

  /// False until the student has both the Vize and the Final (or their
  /// replacements) and every score of this PÇ in them is final.
  bool complete = true;

  void add(double percent, double weight) {
    _weighted += percent * weight;
    _weights += weight;
  }

  /// Weighted mean of the exams' PÇ percentages. An exam without a question
  /// for this PÇ does not count, so its weight goes to the other exam.
  double get percent => _weights == 0 ? 0 : _weighted / _weights;
}

/// Everything the course knows about one student.
class StudentSummary {
  StudentSummary({required this.number, this.name});

  final String number;
  String? name;
  final List<StudentExam> exams = [];

  StudentExam? _of(String kind) =>
      exams.where((e) => e.kind == kind).firstOrNull;

  /// The exam that stands for each weighted slot: Mazeret replaces a missing
  /// Vize, Bütünleme replaces Final.
  Map<String, StudentExam?> get slots => {
    'Vize': _of('Vize') ?? _of('Mazeret'),
    'Final': _of('Bütünleme') ?? _of('Final'),
  };

  /// The exams that count for the course PÇ.
  List<StudentExam> get counted => slots.values.nonNulls.toList();

  /// Course-level PÇ achievement: Vize %40 + Final %60 of each PÇ's percent.
  List<OutcomeTotal> get totals {
    final totals = <String, OutcomeTotal>{};
    final slots = this.slots;
    final missingSlot = slots.values.any((e) => e == null);
    for (final MapEntry(key: slot, value: exam) in slots.entries) {
      if (exam == null) continue;
      for (final o in exam.outcomes) {
        final total = totals.putIfAbsent(o.id, () => OutcomeTotal(o.id));
        // No percentage before the exam's maximum scores are entered.
        if (o.maximumKnown) total.add(o.percent, examWeights[slot]!);
        total.complete &= o.complete && o.maximumKnown;
      }
    }
    for (final total in totals.values) {
      total.complete &= !missingSlot;
    }
    return totals.values.toList()..sort((a, b) => a.id.compareTo(b.id));
  }

  bool get allConfirmed => exams.every((e) => e.confirmed);
}

/// The number written on a sheet: the roster match, else the value read or
/// corrected in the number cell.
String? sheetStudentNumber(SheetResult result) {
  final fromRoster = result.studentNumber;
  if (fromRoster != null && fromRoster.isNotEmpty) return fromRoster;
  final cell = result.cells.where((c) => c.field == 'Numara').firstOrNull;
  final value = cell?.value?.trim();
  return value == null || value.isEmpty ? null : value;
}

/// The course's sheets grouped by student number, sorted by number.
/// Sheets without a readable number are returned separately.
({List<StudentSummary> students, List<StudentExam> unidentified})
courseStudents(CourseEntry course) {
  final byNumber = <String, StudentSummary>{};
  final unidentified = <StudentExam>[];
  final exams = [...course.exams]
    ..sort((a, b) => _kindRank(a).compareTo(_kindRank(b)));
  for (final exam in exams) {
    // Newest first, so a re-scanned sheet replaces the older one.
    final sheets = [...exam.sheets]
      ..sort((a, b) => b.createdAt.compareTo(a.createdAt));
    for (final sheet in sheets) {
      final number = sheetStudentNumber(sheet.result);
      final entry = StudentExam(exam: exam, sheet: sheet);
      if (number == null) {
        unidentified.add(entry);
        continue;
      }
      final student = byNumber.putIfAbsent(
        number,
        () => StudentSummary(number: number),
      );
      student.name ??= sheet.result.studentName;
      if (student.exams.any((e) => e.exam.id == exam.id)) continue;
      // A student sits only one group of an exam kind.
      if (student.exams.any((e) => e.kind == entry.kind)) continue;
      student.exams.add(entry);
    }
  }
  final students = byNumber.values.toList()
    ..sort((a, b) => a.number.compareTo(b.number));
  return (students: students, unidentified: unidentified);
}

int _kindRank(ExamEntry exam) {
  final index = examKindOrder.indexOf(exam.info.examKind ?? '');
  return index < 0 ? examKindOrder.length : index;
}
