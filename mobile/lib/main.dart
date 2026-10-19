import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import 'app_state.dart';
import 'pages/courses_page.dart';
import 'src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  await appState.load();
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
      home: const CoursesPage(),
    );
  }
}
