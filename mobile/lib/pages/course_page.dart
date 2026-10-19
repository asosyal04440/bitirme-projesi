import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../src/rust/api/exam.dart';
import 'exam_page.dart';
import 'exam_setup_page.dart';
import 'students_tab.dart';

/// One course: its exams, each defined by its own Word file.
class CoursePage extends StatelessWidget {
  const CoursePage({super.key, required this.course});

  final CourseEntry course;

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

  Future<void> _openSetup(BuildContext context, Uint8List bytes) async {
    try {
      final info = await inspectExam(docx: bytes);
      if (!context.mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) =>
              ExamSetupPage(course: course, docx: bytes, info: info),
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

  Future<void> _delete(BuildContext context) async {
    final yes = await showShadDialog<bool>(
      context: context,
      builder: (context) => ShadDialog.alert(
        title: Text('${course.code} silinsin mi?'),
        description: Text(
          'Dersin ${course.exams.length} sınavı ve ${course.sheetCount} kâğıdı da silinir.',
        ),
        actions: [
          ShadButton.outline(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Vazgeç'),
          ),
          ShadButton.destructive(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Sil'),
          ),
        ],
      ),
    );
    if (yes != true) return;
    await appState.deleteCourse(course);
    if (context.mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text(course.code, style: theme.textTheme.large),
        actions: [
          ShadIconButton.ghost(
            icon: const Icon(LucideIcons.trash2, size: 18),
            onPressed: () => _delete(context),
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: appState,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (course.name != null) ...[
              Text(course.name!, style: theme.textTheme.h4),
              const SizedBox(height: 16),
            ],
            ShadTabs<String>(
              value: 'exams',
              tabs: [
                ShadTab(
                  value: 'exams',
                  content: _ExamsTab(
                    course: course,
                    onAdd: () => _addExam(context),
                  ),
                  child: const Text('Sınavlar'),
                ),
                ShadTab(
                  value: 'students',
                  content: StudentsTab(course: course),
                  child: const Text('Öğrenciler'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ExamsTab extends StatelessWidget {
  const _ExamsTab({required this.course, required this.onAdd});

  final CourseEntry course;
  final VoidCallback onAdd;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 12),
        ShadButton(
          leading: const Icon(LucideIcons.fileUp, size: 16),
          onPressed: onAdd,
          child: const Text('Word dosyasından sınav ekle'),
        ),
        const SizedBox(height: 16),
        if (course.exams.isEmpty)
          ShadCard(
            title: const Text('Henüz sınav yok'),
            description: const Text(
              'Vize, final veya bütünleme kâğıdının Word dosyasını seçin; '
              'soru–PÇ eşleşmeleri ve tam puanlar otomatik okunur.',
            ),
          )
        else
          for (final exam in course.exams) ...[
            _ExamTile(exam: exam),
            const SizedBox(height: 12),
          ],
      ],
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
        title: Text(exam.label),
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
