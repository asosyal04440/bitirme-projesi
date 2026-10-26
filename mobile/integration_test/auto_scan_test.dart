import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:sinav_okuyucu/app_state.dart';
import 'package:sinav_okuyucu/definition_json.dart';
import 'package:sinav_okuyucu/src/rust/api/camera.dart';
import 'package:sinav_okuyucu/src/rust/api/definition.dart';
import 'package:sinav_okuyucu/src/rust/frb_generated.dart';

/// A clean filled sheet (synthetic) as an RGBA camera frame.
Future<CameraFrame> sheetFrame() async {
  final bytes = (await rootBundle.load('assets/dev/temiz-kagit.jpg')).buffer
      .asUint8List();
  final image = (await (await ui.instantiateImageCodec(
    bytes,
  )).getNextFrame()).image;
  final rgba = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
  return CameraFrame(
    width: image.width,
    height: image.height,
    format: FrameFormat.rgba8888,
    planes: [
      FramePlane(
        bytes: rgba.buffer.asUint8List(),
        rowStride: image.width * 4,
        pixelStride: 4,
      ),
    ],
    rotation: 0,
  );
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await RustLib.init();
    await appState.load();
  });

  testWidgets('the first sheet sets up course and exam, then sheets are read', (
    tester,
  ) async {
    for (final old in appState.courses.where((c) => c.code == 'BİLM374')) {
      await appState.deleteCourse(old);
    }
    final scanner = await appState.scanner();
    scanner.reset();
    final frame = await sheetFrame();

    ExamDefinition? definition;
    final millis = <int>[];
    for (var i = 0; i < 5 && definition == null; i++) {
      final update = await scanner.detectFrame(frame: frame);
      millis.add(update.millis);
      definition = update.definition;
    }
    // ignore: avoid_print
    print('DETECT_FRAMES_MS=$millis');
    expect(definition, isNotNull);
    expect(definition!.courseCode, 'BİLM374');
    expect(definition.examKind, 'Final');
    expect(
      definition.questions.map((q) => '${q.printedId}-${q.outcomes.join()}'),
      ['S1-PÇ2', 'S2-PÇ1', 'S3-PÇ3', 'S4-PÇ2'],
    );
    expect(appState.examFor(definition), isNull);

    // The instructor enters only the maximum scores.
    final exam = await appState.createDetectedExam(
      definitionWith(
        definition,
        outcomes: const {},
        maximums: const {'S1': '20', 'S2': '30', 'S3': '25', 'S4': '25'},
      ),
    );
    expect(appState.courses.map((c) => c.code), contains('BİLM374'));
    expect(appState.examFor(definition)?.id, exam.id);

    final reader = await exam.reader();
    reader.resetScan();
    ScanUpdate? saved;
    for (var i = 0; i < 8 && saved == null; i++) {
      final update = await reader.scanFrame(frame: frame);
      if (update.state == ScanState.saved) saved = update;
    }
    expect(saved, isNotNull);
    expect(
      {for (final c in saved!.sheet!.cells) c.field: c.value},
      {
        'Numara': '03310352098',
        'S1': '4',
        'S2': '14.5',
        'S3': '20',
        'S4': '18',
        'Not': '56.5',
      },
    );

    // The exam survives a restart, stored without a Word file.
    await appState.load();
    final reloaded = appState.examFor(definition)!;
    expect(reloaded.stored.docx, isEmpty);
    expect(reloaded.info.questions.length, 4);
    expect((await reloaded.reader()).examInfo().courseCode, 'BİLM374');

    await appState.deleteCourse(
      appState.courses.singleWhere((c) => c.code == 'BİLM374'),
    );
  });
}
