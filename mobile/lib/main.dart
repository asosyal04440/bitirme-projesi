import 'dart:async';

import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import 'app_state.dart';
import 'pages/courses_page.dart';
import 'src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  await appState.load();
  // The printed-text reader takes a while to prepare on a phone: do it in
  // the background so the first sheet is recognised without waiting.
  unawaited(appState.scanner().then((s) => s.warmUp()).catchError((_) {}));
  runApp(const SinavOkuyucuApp());
}

/// Petrol blue of the İnönü University logo, used as the accent colour.
const _accent = Color(0xFF1C6A7E);

class SinavOkuyucuApp extends StatelessWidget {
  const SinavOkuyucuApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ShadApp(
      title: 'Sınav Okuyucu',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.light,
      theme: ShadThemeData(
        brightness: Brightness.light,
        colorScheme: const ShadSlateColorScheme.light(
          primary: _accent,
          ring: _accent,
        ),
      ),
      darkTheme: ShadThemeData(
        brightness: Brightness.dark,
        colorScheme: const ShadSlateColorScheme.dark(),
      ),
      // Keep every screen above the system navigation bar: on newer Android
      // versions apps are drawn edge to edge, under the bar's buttons.
      builder: (context, child) =>
          SafeArea(top: false, child: child ?? const SizedBox.shrink()),
      home: const CoursesPage(),
    );
  }
}
