import 'src/rust/api/definition.dart';

/// Exam definitions recognised on the sheet are stored as JSON.
Map<String, Object?> definitionToJson(ExamDefinition d) => {
  'course': d.course,
  'courseCode': d.courseCode,
  'courseName': d.courseName,
  'academicYear': d.academicYear,
  'term': d.term,
  'examKind': d.examKind,
  'group': d.group,
  'pageWidth': d.pageWidth,
  'totalLeft': d.totalLeft,
  'totalWidth': d.totalWidth,
  'questions': [
    for (final q in d.questions)
      {
        'printedId': q.printedId,
        'outcomes': q.outcomes,
        'maximum': q.maximum,
        'left': q.left,
        'width': q.width,
      },
  ],
};

ExamDefinition definitionFromJson(Map<String, dynamic> j) => ExamDefinition(
  course: j['course'] as String?,
  courseCode: j['courseCode'] as String?,
  courseName: j['courseName'] as String?,
  academicYear: j['academicYear'] as String?,
  term: j['term'] as String?,
  examKind: j['examKind'] as String?,
  group: j['group'] as String?,
  pageWidth: j['pageWidth'] as int,
  totalLeft: j['totalLeft'] as int,
  totalWidth: j['totalWidth'] as int,
  questions: [
    for (final q in j['questions'] as List)
      DefinedQuestion(
        printedId: q['printedId'] as String,
        outcomes: List<String>.from(q['outcomes'] as List),
        maximum: q['maximum'] as String?,
        left: q['left'] as int,
        width: q['width'] as int,
      ),
  ],
);

/// A copy with the instructor's outcomes and maximum scores per question.
ExamDefinition definitionWith(
  ExamDefinition d, {
  String? courseCode,
  String? examKind,
  String? group,
  required Map<String, List<String>> outcomes,
  required Map<String, String> maximums,
}) => ExamDefinition(
  course: d.course,
  courseCode: courseCode ?? d.courseCode,
  courseName: d.courseName,
  academicYear: d.academicYear,
  term: d.term,
  examKind: examKind ?? d.examKind,
  group: group ?? d.group,
  pageWidth: d.pageWidth,
  totalLeft: d.totalLeft,
  totalWidth: d.totalWidth,
  questions: [
    for (final q in d.questions)
      DefinedQuestion(
        printedId: q.printedId,
        outcomes: outcomes[q.printedId] ?? q.outcomes,
        maximum: maximums[q.printedId] ?? q.maximum,
        left: q.left,
        width: q.width,
      ),
  ],
);
