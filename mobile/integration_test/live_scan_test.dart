import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:sinav_okuyucu/app_state.dart';
import 'package:sinav_okuyucu/src/rust/api/camera.dart';
import 'package:sinav_okuyucu/src/rust/frb_generated.dart';

Future<Uint8List> asset(String path) async =>
    (await rootBundle.load(path)).buffer.asUint8List();

/// The bundled sample sheet photo as an RGBA camera frame.
Future<CameraFrame> sheetFrame() async {
  final codec = await ui.instantiateImageCodec(
    await asset('assets/dev/ornek-kagit.jpg'),
  );
  final image = (await codec.getNextFrame()).image;
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

CameraFrame emptyFrame(int width, int height) => CameraFrame(
  width: width,
  height: height,
  format: FrameFormat.rgba8888,
  planes: [
    FramePlane(
      bytes: Uint8List(width * height * 4)
        ..fillRange(0, width * height * 4, 90),
      rowStride: width * 4,
      pixelStride: 4,
    ),
  ],
  rotation: 0,
);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await RustLib.init();
    await appState.load();
  });

  testWidgets('a sheet held in front of the camera is saved once, quickly', (
    tester,
  ) async {
    for (final old in appState.courses.where((c) => c.code == 'BİLM374')) {
      await appState.deleteCourse(old);
    }
    final course = await appState.createCourse(code: 'BİLM374');
    final exam = await appState.createExam(
      course: course,
      docx: await asset('assets/dev/ornek-sinav.docx'),
      maximums: {'S3': '25', 'S4': '25'},
      roster: const [],
    );
    final reader = await exam.reader();
    reader.resetScan();

    final sheet = await sheetFrame();
    final empty = emptyFrame(sheet.width, sheet.height);
    final frames = [empty, for (var i = 0; i < 8; i++) sheet, empty];
    final states = <ScanState>[];
    final millis = <int>[];
    for (final frame in frames) {
      final update = await reader.scanFrame(frame: frame);
      states.add(update.state);
      millis.add(update.millis);
    }
    debugPrintFrames(states, millis);

    expect(states.first, ScanState.searching);
    expect(states.where((s) => s == ScanState.saved).length, 1);
    expect(states, contains(ScanState.alreadySaved));
    // From the sheet coming into view until it is saved.
    final savedAt = states.indexOf(ScanState.saved);
    final toSave = millis.sublist(1, savedAt + 1).reduce((a, b) => a + b);
    // ignore: avoid_print
    print('LIVE_TIME_TO_SAVE_MS=$toSave');

    await appState.deleteCourse(course);
  });
}

void debugPrintFrames(List<ScanState> states, List<int> millis) {
  for (var i = 0; i < states.length; i++) {
    // ignore: avoid_print
    print('LIVE_FRAME $i ${states[i].name} ${millis[i]} ms');
  }
}
