import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../app_state.dart';
import '../camera_feed.dart';
import '../src/rust/api/scanner.dart';
import 'detected_exam_page.dart';
import 'live_scan_page.dart';

/// Starting point without any set-up: the camera opens, the instructor shows
/// the first sheet and the exam is recognised from it (course, kind, the
/// Soru-PÇ table). A known exam goes straight to scanning; a new one is
/// confirmed once.
class AutoScanPage extends StatefulWidget {
  const AutoScanPage({super.key});

  @override
  State<AutoScanPage> createState() => _AutoScanPageState();
}

class _AutoScanPageState extends State<AutoScanPage>
    with WidgetsBindingObserver {
  final _feed = CameraFeed();
  DetectUpdate? _update;
  bool _done = false;

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
    _feed.stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _feed.stop();
    } else if (state == AppLifecycleState.resumed &&
        _feed.controller == null &&
        !_done) {
      _start();
    }
  }

  Future<void> _start() async {
    final scanner = await appState.scanner();
    scanner.reset();
    await _feed.start(
      (frame) => scanner.detectFrame(frame: frame).then(_handle),
    );
    if (mounted) setState(() {});
  }

  Future<void> _handle(DetectUpdate update) async {
    if (!mounted || _done) return;
    setState(() => _update = update);
    final definition = update.definition;
    if (definition == null) return;
    _done = true;
    HapticFeedback.mediumImpact();
    await _feed.stop();
    if (!mounted) return;
    final known = appState.examFor(definition);
    await Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => known != null
            ? LiveScanPage(exam: known)
            : DetectedExamPage(definition: definition),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final camera = _feed.controller;
    final update = _update;
    final status = update == null || !update.found
        ? 'İlk kâğıdı gösterin: başlık ve Soru–PÇ tablosu kadrajda olsun'
        : 'Sınav tanınıyor…';
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          fit: StackFit.expand,
          children: [
            if (camera != null && _feed.ready)
              Center(
                child: CameraPreview(
                  camera,
                  child: CustomPaint(
                    painter: TableOutlinePainter(
                      table: update?.table ?? const [],
                      turn: _feed.deviceTurn,
                      color: Colors.white,
                    ),
                  ),
                ),
              )
            else
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Text(
                    _feed.error ?? 'Kamera açılıyor…',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white70),
                  ),
                ),
              ),
            Align(
              alignment: Alignment.topCenter,
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(8, 8, 12, 0),
                  child: Row(
                    children: [
                      IconButton(
                        icon: const Icon(LucideIcons.x, color: Colors.white),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                      const Text(
                        'Kâğıt okut',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Align(
              alignment: Alignment.bottomCenter,
              child: SafeArea(
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
                        status,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 6),
                      const Text(
                        'Ders, sınav ve soru–PÇ eşleşmeleri kâğıttan okunur; '
                        'Word dosyası gerekmez.',
                        style: TextStyle(color: Colors.white70, fontSize: 13),
                      ),
                    ],
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
