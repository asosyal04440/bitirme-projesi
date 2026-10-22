import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'app_state.dart';
import 'src/rust/api/reader.dart';

/// Writes the exam's Excel report (grades, PÇ and question analysis) to the
/// app's documents folder and returns the file. Called after every sheet
/// saved during live scanning, so the file is always up to date.
Future<File> writeExamExcel(ExamEntry exam) async {
  final reader = await exam.reader();
  final bytes = await reader.exportExcel(
    sheets: [
      for (final record in exam.sheets.reversed)
        ReportSheet(
          studentNumber: record.result.studentNumber,
          studentName: record.result.studentName,
          values: [
            for (final cell in record.result.cells)
              FieldValue(
                field: cell.field,
                // Unconfirmed sheets report only values the app accepted.
                value: record.confirmed || cell.accepted ? cell.value : null,
                region: cell.region,
              ),
          ],
          outcomes: record.result.outcomes,
          confirmed: record.confirmed,
        ),
    ],
  );
  final directory = Directory(
    '${(await getApplicationDocumentsDirectory()).path}/excel',
  );
  await directory.create(recursive: true);
  final file = File('${directory.path}/${excelFileName(exam)}');
  await file.writeAsBytes(bytes, flush: true);
  return file;
}

/// "BİLM374-Final-grup-A.xlsx"
String excelFileName(ExamEntry exam) {
  final name = [
    exam.info.courseCode ?? 'sinav',
    exam.info.examKind,
    if (exam.info.group != null) 'grup-${exam.info.group}',
  ].whereType<String>().join('-');
  return '$name.xlsx';
}
