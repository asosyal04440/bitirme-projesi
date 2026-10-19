import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import 'course_page.dart';

/// Home: the instructor's courses. Exams and sheets are kept under them.
class CoursesPage extends StatelessWidget {
  const CoursesPage({super.key});

  Future<void> _addCourse(BuildContext context) async {
    final course = await showShadDialog<CourseEntry>(
      context: context,
      builder: (_) => const _AddCourseDialog(),
    );
    if (course == null || !context.mounted) return;
    await Navigator.of(
      context,
    ).push(MaterialPageRoute<void>(builder: (_) => CoursePage(course: course)));
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
              Text('Dersler', style: theme.textTheme.h2),
              const SizedBox(height: 4),
              Text(
                'Dersi ekleyin, sınavını Word dosyasından tanımlayın, kâğıtları tarayın.',
                style: theme.textTheme.muted,
              ),
              const SizedBox(height: 20),
              ShadButton(
                width: double.infinity,
                leading: const Icon(LucideIcons.plus, size: 16),
                onPressed: () => _addCourse(context),
                child: const Text('Ders ekle'),
              ),
              const SizedBox(height: 20),
              if (appState.courses.isEmpty)
                ShadCard(
                  title: const Text('Henüz ders yok'),
                  description: const Text(
                    'Ders kodunu girerek başlayın, örneğin BİLM374.',
                  ),
                )
              else
                for (final course in appState.courses) ...[
                  _CourseTile(course: course),
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

class _CourseTile extends StatelessWidget {
  const _CourseTile({required this.course});

  final CourseEntry course;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.of(context).push(
        MaterialPageRoute<void>(builder: (_) => CoursePage(course: course)),
      ),
      child: ShadCard(
        title: Text(course.code),
        description: course.name == null ? null : Text(course.name!),
        trailing: const Icon(LucideIcons.chevronRight, size: 18),
        child: Padding(
          padding: const EdgeInsets.only(top: 12),
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ShadBadge.secondary(child: Text('${course.exams.length} sınav')),
              ShadBadge.secondary(child: Text('${course.sheetCount} kâğıt')),
              if (course.pendingCount > 0)
                ShadBadge.outline(
                  child: Text('${course.pendingCount} kâğıt kontrol bekliyor'),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _AddCourseDialog extends StatefulWidget {
  const _AddCourseDialog();

  @override
  State<_AddCourseDialog> createState() => _AddCourseDialogState();
}

class _AddCourseDialogState extends State<_AddCourseDialog> {
  final _code = TextEditingController();
  final _name = TextEditingController();
  String? _error;
  bool _busy = false;

  @override
  void dispose() {
    _code.dispose();
    _name.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (normalizeCourseCode(_code.text).isEmpty) {
      setState(() => _error = 'Ders kodunu girin.');
      return;
    }
    setState(() => _busy = true);
    try {
      final course = await appState.createCourse(
        code: _code.text,
        name: _name.text,
      );
      if (mounted) Navigator.of(context).pop(course);
    } on StateError catch (error) {
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return ShadDialog(
      title: const Text('Ders ekle'),
      description: const Text('Sınav kâğıtları bu dersin altında saklanır.'),
      actions: [
        ShadButton.outline(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Vazgeç'),
        ),
        ShadButton(
          enabled: !_busy,
          onPressed: _save,
          child: const Text('Ekle'),
        ),
      ],
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Ders kodu', style: theme.textTheme.small),
            const SizedBox(height: 6),
            ShadInput(
              controller: _code,
              autofocus: true,
              placeholder: const Text('BİLM374'),
              textCapitalization: TextCapitalization.characters,
              inputFormatters: const [_CourseCodeFormatter()],
              onChanged: (_) {
                if (_error != null) setState(() => _error = null);
              },
              onSubmitted: (_) => _save(),
            ),
            const SizedBox(height: 12),
            Text('Ders adı (isteğe bağlı)', style: theme.textTheme.small),
            const SizedBox(height: 6),
            ShadInput(
              controller: _name,
              placeholder: const Text('Yapay Zeka'),
              textCapitalization: TextCapitalization.words,
              onSubmitted: (_) => _save(),
            ),
            if (_error != null) ...[
              const SizedBox(height: 10),
              Text(
                _error!,
                style: theme.textTheme.small.copyWith(
                  color: theme.colorScheme.destructive,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Shows the code in capitals while typing, with Turkish i → İ.
class _CourseCodeFormatter extends TextInputFormatter {
  const _CourseCodeFormatter();

  @override
  TextEditingValue formatEditUpdate(
    TextEditingValue oldValue,
    TextEditingValue newValue,
  ) => newValue.copyWith(
    text: newValue.text.replaceAll('i', 'İ').replaceAll('ı', 'I').toUpperCase(),
  );
}
