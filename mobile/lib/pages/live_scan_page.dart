import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../app_state.dart';
import '../excel_export.dart';
import '../src/rust/api/camera.dart';
import '../src/rust/api/reader.dart';
import '../storage.dart';
import 'sheet_review_page.dart';

/// Sheets are read straight from the camera, one after another: the
/// instructor shows a sheet, the Soru-PÇ table is found and read, the sheet is
/// saved and a green tick confirms it; the camera stays on for the next one.
class LiveScanPage extends StatefulWidget {
  const LiveScanPage({super.key, required this.exam});

  final ExamEntry exam;

  @override
  State<LiveScanPage> createState() => _LiveScanPageState();
}

class _LiveScanPageState extends State<LiveScanPage>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  static const _green = Color(0xFF16A34A);
  static const _amber = Color(0xFFD97706);

  CameraController? _camera;
  ExamReader? _reader;
  String? _error;
  bool _busy = false;

  ScanUpdate? _update;
  final List<SheetRecord> _saved = [];

  /// The tick shown for a moment after a sheet is saved.
  late final AnimationController _tick = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  );
  bool _tickVerified = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WakelockPlus.enable();
    _start();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    WakelockPlus.disable();
    _tick.dispose();
    _stopCamera();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed && _camera == null) {
      _start();
    }
  }

  Future<void> _start() async {
    try {
      final reader = await widget.exam.reader();
      reader.resetScan();
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
      if (!mounted) {
        await camera.dispose();
        return;
      }
      setState(() {
        _reader = reader;
        _camera = camera;
        _error = null;
      });
      await camera.startImageStream(_onFrame);
    } on CameraException catch (error) {
      setState(
        () => _error = error.code == 'CameraAccessDenied'
            ? 'Kamera izni verilmedi. Ayarlar’dan izin verip tekrar deneyin.'
            : 'Kamera açılamadı: ${error.description ?? error.code}',
      );
    } catch (error) {
      setState(() => _error = 'Kamera açılamadı: $error');
    }
  }

  Future<void> _stopCamera() async {
    final camera = _camera;
    _camera = null;
    if (camera == null) return;
    if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
    await camera.dispose();
  }

  /// Clockwise rotation that turns the sensor image upright, also when the
  /// phone is held sideways (the screen itself stays in portrait).
  int get _rotation {
    final camera = _camera!;
    final device = switch (camera.value.deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };
    return (camera.description.sensorOrientation - device + 360) % 360;
  }

  /// How far the phone is turned from portrait, to draw the outline found in
  /// the upright frame onto the portrait preview.
  int get _deviceTurn =>
      (_camera!.description.sensorOrientation - _rotation + 360) % 360;

  void _onFrame(CameraImage image) {
    final reader = _reader;
    if (_busy || reader == null || _camera == null) return;
    _busy = true;
    final frame = CameraFrame(
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
      rotation: _rotation,
    );
    reader
        .scanFrame(frame: frame)
        .then(_handle)
        .catchError((Object error) => debugPrint('scan failed: $error'))
        .whenComplete(() => _busy = false);
  }

  Future<void> _handle(ScanUpdate update) async {
    if (!mounted) return;
    setState(() => _update = update);
    final sheet = update.sheet;
    if (update.state != ScanState.saved || sheet == null) return;

    final record = SheetRecord(
      id: sheet.sheetId,
      createdAt: DateTime.now(),
      confirmed: update.verified,
      result: sheet,
    );
    HapticFeedback.heavyImpact();
    SystemSound.play(SystemSoundType.click);
    setState(() {
      _saved.insert(0, record);
      _tickVerified = update.verified;
    });
    _tick.forward(from: 0);
    await appState.saveSheet(widget.exam, record);
    unawaited(writeExamExcel(widget.exam));
  }

  Future<void> _shareExcel() async {
    final file = await writeExamExcel(widget.exam);
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(file.path)],
        subject: '${widget.exam.title} – PÇ raporu',
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          fit: StackFit.expand,
          children: [
            if (camera != null && camera.value.isInitialized)
              Center(
                child: CameraPreview(
                  camera,
                  child: CustomPaint(
                    painter: _TablePainter(
                      table: _update?.table ?? const [],
                      turn: _deviceTurn,
                      color:
                          _update?.state == ScanState.saved ||
                              _update?.state == ScanState.alreadySaved
                          ? _green
                          : Colors.white,
                    ),
                  ),
                ),
              )
            else
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Text(
                    _error ?? 'Kamera açılıyor…',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white70),
                  ),
                ),
              ),
            _TopBar(
              title:
                  '${widget.exam.info.courseCode ?? widget.exam.title} · '
                  '${widget.exam.label}',
              count: _saved.length,
              onExcel: _saved.isEmpty && widget.exam.sheets.isEmpty
                  ? null
                  : _shareExcel,
            ),
            Align(
              alignment: Alignment.center,
              child: _Tick(animation: _tick, verified: _tickVerified),
            ),
            Align(
              alignment: Alignment.bottomCenter,
              child: _StatusPanel(
                update: _update,
                last: _saved.isEmpty ? null : _saved.first,
                onOpenLast: _saved.isEmpty
                    ? null
                    : () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => SheetReviewPage(
                            exam: widget.exam,
                            record: widget.exam.sheets.firstWhere(
                              (s) => s.id == _saved.first.id,
                              orElse: () => _saved.first,
                            ),
                          ),
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// The Soru-PÇ table outline over the preview.
class _TablePainter extends CustomPainter {
  _TablePainter({required this.table, required this.turn, required this.color});

  /// `[x0, y0, … x3, y3]` normalised to the upright frame.
  final List<double> table;

  /// Clockwise degrees from the upright frame to the portrait preview.
  final int turn;
  final Color color;

  Offset _toPreview(double u, double v, Size size) {
    final (x, y) = switch (turn) {
      90 => (1 - v, u),
      180 => (1 - u, 1 - v),
      270 => (v, 1 - u),
      _ => (u, v),
    };
    return Offset(x * size.width, y * size.height);
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (table.length != 8) return;
    final path = Path();
    for (var i = 0; i < 4; i++) {
      final p = _toPreview(table[2 * i], table[2 * i + 1], size);
      i == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
    }
    path.close();
    canvas.drawPath(
      path,
      Paint()
        ..color = color.withValues(alpha: 0.18)
        ..style = PaintingStyle.fill,
    );
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..strokeJoin = StrokeJoin.round,
    );
  }

  @override
  bool shouldRepaint(_TablePainter old) =>
      !listEquals(old.table, table) || old.color != color || old.turn != turn;
}

class _TopBar extends StatelessWidget {
  const _TopBar({required this.title, required this.count, this.onExcel});

  final String title;
  final int count;
  final VoidCallback? onExcel;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(8, 8, 12, 0),
        child: Row(
          children: [
            IconButton(
              icon: const Icon(LucideIcons.x, color: Colors.white),
              onPressed: () => Navigator.of(context).pop(),
            ),
            Expanded(
              child: Text(
                title,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            _Pill(text: '$count kâğıt'),
            const SizedBox(width: 8),
            ShadButton.secondary(
              size: ShadButtonSize.sm,
              enabled: onExcel != null,
              leading: const Icon(LucideIcons.fileSpreadsheet, size: 16),
              onPressed: onExcel,
              child: const Text('Excel'),
            ),
          ],
        ),
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 13),
      ),
    );
  }
}

/// A large tick that pops up and fades out after a sheet is saved.
class _Tick extends StatelessWidget {
  const _Tick({required this.animation, required this.verified});

  final Animation<double> animation;
  final bool verified;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: AnimatedBuilder(
        animation: animation,
        builder: (context, _) {
          final t = animation.value;
          if (t == 0 || t == 1) return const SizedBox.shrink();
          final scale = Curves.easeOutBack.transform((t * 3).clamp(0.0, 1.0));
          final opacity = t < 0.7 ? 1.0 : 1 - (t - 0.7) / 0.3;
          return Opacity(
            opacity: opacity,
            child: Transform.scale(
              scale: 0.6 + 0.4 * scale,
              child: Container(
                width: 132,
                height: 132,
                decoration: BoxDecoration(
                  color: verified
                      ? _LiveScanPageState._green
                      : _LiveScanPageState._amber,
                  shape: BoxShape.circle,
                  boxShadow: const [
                    BoxShadow(color: Colors.black38, blurRadius: 24),
                  ],
                ),
                child: Icon(
                  verified ? LucideIcons.check : LucideIcons.triangleAlert,
                  color: Colors.white,
                  size: 72,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _StatusPanel extends StatelessWidget {
  const _StatusPanel({required this.update, this.last, this.onOpenLast});

  final ScanUpdate? update;
  final SheetRecord? last;
  final VoidCallback? onOpenLast;

  String get _status => switch (update?.state) {
    ScanState.reading => 'Okunuyor…',
    ScanState.saved =>
      update!.verified ? 'Okundu ve kaydedildi' : 'Kaydedildi, kontrol edin',
    ScanState.alreadySaved => 'Bu kâğıt kaydedildi. Sıradakini gösterin.',
    _ => 'Kâğıdı gösterin: başlık ve Soru–PÇ tablosu kadrajda olsun',
  };

  @override
  Widget build(BuildContext context) {
    final last = this.last;
    return SafeArea(
      child: Container(
        margin: const EdgeInsets.all(12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.72),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _status,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            if (last != null) ...[
              const SizedBox(height: 10),
              GestureDetector(
                onTap: onOpenLast,
                child: Row(
                  children: [
                    Icon(
                      last.confirmed
                          ? LucideIcons.circleCheck
                          : LucideIcons.circleAlert,
                      size: 18,
                      color: last.confirmed
                          ? _LiveScanPageState._green
                          : _LiveScanPageState._amber,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _summary(last.result),
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                          fontFeatures: [FontFeature.tabularFigures()],
                        ),
                      ),
                    ),
                    const Icon(
                      LucideIcons.chevronRight,
                      size: 18,
                      color: Colors.white54,
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// "04451095728 · S1 13 · S2 23 · S3 23 · S4 11 · Not 70"
  static String _summary(SheetResult sheet) => [
    for (final cell in sheet.cells)
      cell.field == 'Numara'
          ? (sheet.studentName ?? cell.value ?? '—')
          : '${cell.field} ${cell.value ?? '—'}',
  ].join(' · ');
}
