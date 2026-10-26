import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'src/rust/api/camera.dart';

/// The back camera's live frames, handed over one at a time: a frame
/// arriving while the previous one is still being processed is skipped.
class CameraFeed {
  CameraController? controller;

  /// Turkish message when the camera could not be opened.
  String? error;
  bool _busy = false;

  bool get ready => controller?.value.isInitialized ?? false;

  /// Opens the camera and calls [onFrame] for frame after frame.
  Future<void> start(Future<void> Function(CameraFrame) onFrame) async {
    try {
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final camera = CameraController(
        back,
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );
      await camera.initialize();
      await camera.lockCaptureOrientation(DeviceOrientation.portraitUp);
      controller = camera;
      error = null;
      await camera.startImageStream((image) {
        if (_busy || controller == null) return;
        _busy = true;
        onFrame(_frame(image))
            .catchError((Object e) => debugPrint('frame failed: $e'))
            .whenComplete(() => _busy = false);
      });
    } on CameraException catch (e) {
      error = e.code == 'CameraAccessDenied'
          ? 'Kamera izni verilmedi. Ayarlar’dan izin verip tekrar deneyin.'
          : 'Kamera açılamadı: ${e.description ?? e.code}';
    } catch (e) {
      error = 'Kamera açılamadı: $e';
    }
  }

  Future<void> stop() async {
    final camera = controller;
    controller = null;
    if (camera == null) return;
    if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
    await camera.dispose();
  }

  /// Clockwise rotation that turns the sensor image upright, also when the
  /// phone is held sideways (the screen itself stays in portrait).
  int get rotation {
    final camera = controller!;
    final device = switch (camera.value.deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };
    return (camera.description.sensorOrientation - device + 360) % 360;
  }

  /// How far the phone is turned from portrait, to draw what was found in
  /// the upright frame onto the portrait preview.
  int get deviceTurn =>
      (controller!.description.sensorOrientation - rotation + 360) % 360;

  CameraFrame _frame(CameraImage image) => CameraFrame(
    width: image.width,
    height: image.height,
    format: image.format.group == ImageFormatGroup.bgra8888
        ? FrameFormat.bgra8888
        : FrameFormat.yuv420,
    planes: [
      for (final plane in image.planes)
        FramePlane(
          bytes: plane.bytes,
          rowStride: plane.bytesPerRow,
          pixelStride: plane.bytesPerPixel ?? 1,
        ),
    ],
    rotation: rotation,
  );
}
