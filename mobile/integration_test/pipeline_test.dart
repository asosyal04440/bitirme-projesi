import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:sinav_okuyucu/app_state.dart';
import 'package:sinav_okuyucu/pages/sheet_review_page.dart';
import 'package:sinav_okuyucu/src/rust/api/exam.dart';
import 'package:sinav_okuyucu/src/rust/api/reader.dart';
import 'package:sinav_okuyucu/src/rust/frb_generated.dart';
import 'package:sinav_okuyucu/storage.dart';

Future<Uint8List> asset(String path) async =>
    (await rootBundle.load(path)).buffer.asUint8List();

/// The true values of the bundled synthetic sample sheet.
const truth = {
  'Numara': '04451095728',
  'S1': '13',
  'S2': '23',
  'S3': '23',
  'S4': '11',
  'Not': '70',
};

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await RustLib.init();
    await appState.load();
  });

  testWidgets('scan, correct, confirm, persist and export on the device', (
    tester,
  ) async {
    final docx = await asset('assets/dev/ornek-sinav.docx');
    final info = await inspectExam(docx: docx);
    expect(info.courseCode, 'BİLM374');
    expect(info.questions.map((q) => q.id), ['S1', 'S2', 'S3', 'S4']);
    expect(info.ready, isFalse); // S3 and S4 points are not in the file

    for (final old in appState.courses.where((c) => c.code == 'BİLM374')) {
      await appState.deleteCourse(old);
    }
    final course = await appState.createCourse(code: 'bilm 374');
    expect(course.code, 'BİLM374');
    final other = await appState.createCourse(code: 'TEST101');
    await expectLater(
      appState.createExam(
        course: other,
        docx: docx,
        maximums: {'S3': '25', 'S4': '25'},
        roster: const [],
      ),
      throwsA(isA<StateError>()), // the Word file belongs to BİLM374
    );
    await appState.deleteCourse(other);

    final exam = await appState.createExam(
      course: course,
      docx: docx,
      maximums: {'S3': '25', 'S4': '25'},
      roster: const [
        RosterStudent(number: '04451095728', name: 'Zeynep Demir'),
      ],
    );
    final reader = await exam.reader();
    final watch = Stopwatch()..start();
    final result = await reader.processPhoto(
      sheetId: 'ornek',
      photo: await asset('assets/dev/ornek-kagit.jpg'),
    );
    debugPrint('PIPELINE_MS=${watch.elapsedMilliseconds}');
    expect(result.pageFound, isTrue);
    expect(result.cells.length, 6);
    expect(result.studentName, 'Zeynep Demir');

    // The instructor types the true values into the review screen.
    await tester.pumpWidget(
      ShadApp(
        home: SheetReviewPage(
          exam: exam,
          fresh: true,
          record: SheetRecord(
            id: result.sheetId,
            createdAt: DateTime.now(),
            confirmed: false,
            result: result,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    debugPrint('REVIEW_SCREEN_READY');
    await Future<void>.delayed(const Duration(seconds: 5));
    // Rows are built lazily: scroll each field into view before typing.
    for (final cell in result.cells) {
      final label = find.text(cell.field).last;
      await tester.scrollUntilVisible(
        label,
        120,
        scrollable: find.byType(Scrollable).first,
      );
      final row = find.ancestor(of: label, matching: find.byType(Row)).first;
      await tester.enterText(
        find.descendant(of: row, matching: find.byType(EditableText)),
        truth[cell.field]!,
      );
    }
    await tester.pump();
    await tester.ensureVisible(find.text('Onayla ve kaydet'));
    await tester.tap(find.text('Onayla ve kaydet'));
    await tester.pumpAndSettle(const Duration(seconds: 1));

    final saved = exam.sheets.singleWhere((s) => s.id == result.sheetId);
    expect(saved.confirmed, isTrue);
    expect(saved.result.issues, isEmpty);
    expect(saved.result.outcomes.every((o) => o.complete), isTrue);
    expect(saved.result.cells.every((c) => c.cropPng.isNotEmpty), isTrue);

    // Everything survives a restart of the app state.
    await appState.load();
    final reloadedCourse = appState.courses.singleWhere(
      (c) => c.code == 'BİLM374',
    );
    expect(reloadedCourse.name, 'YAPAY ZEKA'); // taken from the Word file
    final reloaded = reloadedCourse.exams.singleWhere((e) => e.id == exam.id);
    final reloadedSheet = reloaded.sheets.singleWhere(
      (s) => s.id == result.sheetId,
    );
    expect(reloadedSheet.confirmed, isTrue);
    expect(
      reloadedSheet.result.cells.firstWhere((c) => c.field == 'S4').value,
      '11',
    );
    expect(
      reloadedSheet.result.cells.every((c) => c.cropPng.isNotEmpty),
      isTrue,
    );

    final excel = await (await reloaded.reader()).exportExcel(
      sheets: [
        ReportSheet(
          studentNumber: reloadedSheet.result.studentNumber,
          studentName: reloadedSheet.result.studentName,
          values: [
            for (final c in reloadedSheet.result.cells)
              FieldValue(field: c.field, value: c.value, region: c.region),
          ],
          outcomes: reloadedSheet.result.outcomes,
          confirmed: true,
        ),
      ],
    );
    expect(excel.sublist(0, 2), [0x50, 0x4B]); // "PK": a zip/xlsx file

    await appState.deleteCourse(reloadedCourse);
    expect(appState.exams.any((e) => e.id == exam.id), isFalse);
  });
}
