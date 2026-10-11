import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../src/rust/api/exam.dart';
import 'exam_page.dart';
import 'exam_setup_page.dart';

class ExamsPage extends StatelessWidget {
  const ExamsPage({super.key});

  Future<void> _addExam(BuildContext context) async {
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['docx'],
    );
    if (picked.isEmpty) return;
    final bytes = await picked.single.readAsBytes();
    if (!context.mounted) return;
    await _openSetup(context, bytes);
  }

  /// The bundled synthetic sample exam, for trying the app without paper.
  Future<void> _addSample(BuildContext context) async {
    final bytes = (await rootBundle.load('assets/dev/ornek-sinav.docx')).buffer
        .asUint8List();
    if (!context.mounted) return;
    await _openSetup(context, bytes);
  }

  Future<void> _openSetup(BuildContext context, Uint8List bytes) async {
    try {
      final info = await inspectExam(docx: bytes);
      if (!context.mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ExamSetupPage(docx: bytes, info: info),
        ),
      );
    } catch (error) {
      if (!context.mounted) return;
      ShadToaster.of(context).show(
        ShadToast.destructive(
          title: const Text('Word dosyası okunamadı'),
          description: Text('$error'),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      body: SafeArea(
        child: ListenableBuilder(
          listenable: appState,
          builder: (context, _) => ListView(
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 24),
            children: [
              Text('Sınavlar', style: theme.textTheme.h2),
              const SizedBox(height: 4),
              Text(
                'Sınavı Word dosyasından ekleyin, kâğıtları tarayın, PÇ başarısını görün.',
                style: theme.textTheme.muted,
              ),
              const SizedBox(height: 20),
              ShadButton(
                width: double.infinity,
                leading: const Icon(LucideIcons.fileUp, size: 16),
                onPressed: () => _addExam(context),
                child: const Text('Word dosyasından sınav ekle'),
              ),
              const SizedBox(height: 8),
              ShadButton.ghost(
                width: double.infinity,
                onPressed: () => _addSample(context),
                child: const Text('Örnek sınavla dene'),
              ),
              const SizedBox(height: 20),
              if (appState.exams.isEmpty)
                ShadCard(
                  title: const Text('Henüz sınav yok'),
                  description: const Text(
                    'Sınav kâğıdının Word dosyasını seçtiğinizde soru–PÇ eşleşmeleri ve tam puanlar otomatik okunur.',
                  ),
                )
              else
                for (final exam in appState.exams) ...[
                  _ExamTile(exam: exam),
                  const SizedBox(height: 12),
                ],
              if (appState.correctionCount > 0)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    'Cihazda ${appState.correctionCount} düzeltilmiş hücre saklanıyor; '
                    'tanıma modelini iyileştirmek için kullanılabilir.',
                    style: theme.textTheme.muted,
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ExamTile extends StatelessWidget {
  const _ExamTile({required this.exam});

  final ExamEntry exam;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final pending = exam.sheets.where((s) => !s.confirmed).length;
    return GestureDetector(
      onTap: () => Navigator.of(context)
          .push(MaterialPageRoute<void>(builder: (_) => ExamPage(exam: exam))),
      child: ShadCard(
        title: Text(exam.title),
        description: Text(exam.subtitle),
        trailing: const Icon(LucideIcons.chevronRight, size: 18),
        child: Padding(
          padding: const EdgeInsets.only(top: 12),
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ShadBadge.secondary(child: Text('${exam.sheets.length} kâğıt')),
              ShadBadge.secondary(
                child: Text('${exam.info.questions.length} soru'),
              ),
              if (pending > 0)
                ShadBadge.outline(
                  child: Text('$pending kâğıt kontrol bekliyor'),
                ),
              if (exam.rosterSize == 0)
                Text('Öğrenci listesi yok', style: theme.textTheme.muted),
            ],
          ),
        ),
      ),
    );
  }
}
