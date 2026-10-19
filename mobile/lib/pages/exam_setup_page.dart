import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../src/rust/api/exam.dart';
import '../src/rust/api/reader.dart';
import 'exam_page.dart';

/// Shows what was read from the Word file and asks only for what is missing.
class ExamSetupPage extends StatefulWidget {
  const ExamSetupPage({
    super.key,
    required this.course,
    required this.docx,
    required this.info,
  });

  final CourseEntry course;
  final Uint8List docx;
  final ExamInfo info;

  @override
  State<ExamSetupPage> createState() => _ExamSetupPageState();
}

class _ExamSetupPageState extends State<ExamSetupPage> {
  final _maximums = <String, String>{};
  List<RosterStudent> _roster = const [];
  String? _rosterFile;
  bool _busy = false;

  late final String? _mismatch = courseMismatch(widget.course, widget.info);

  bool get _complete => widget.info.questions.every(
    (q) =>
        q.maximumScore != null || (_maximums[q.id]?.trim().isNotEmpty ?? false),
  );

  Future<void> _pickRoster() async {
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['csv', 'txt'],
    );
    if (picked.isEmpty) return;
    final file = picked.single;
    final text = utf8.decode(await file.readAsBytes(), allowMalformed: true);
    setState(() {
      _roster = parseRoster(text);
      _rosterFile = file.name;
    });
  }

  Future<void> _create() async {
    setState(() => _busy = true);
    try {
      final exam = await appState.createExam(
        course: widget.course,
        docx: widget.docx,
        maximums: {
          for (final entry in _maximums.entries)
            if (entry.value.trim().isNotEmpty) entry.key: entry.value.trim(),
        },
        roster: _roster,
      );
      if (!mounted) return;
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(builder: (_) => ExamPage(exam: exam)),
      );
    } catch (error) {
      if (!mounted) return;
      ShadToaster.of(context).show(
        ShadToast.destructive(
          title: const Text('Sınav oluşturulamadı'),
          description: Text('$error'),
        ),
      );
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final info = widget.info;
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text(
          '${widget.course.code} · yeni sınav',
          style: theme.textTheme.large,
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ShadCard(
            title: Text(
              info.courseName ?? info.course ?? 'Ders bilgisi bulunamadı',
            ),
            description: Text(
              [
                info.courseCode,
                info.examKind,
                if (info.group != null) 'Grup ${info.group}',
                [info.academicYear, info.term].whereType<String>().join(' '),
                info.date,
              ].whereType<String>().where((s) => s.isNotEmpty).join(' · '),
            ),
          ),
          const SizedBox(height: 16),
          if (_mismatch != null) ...[
            ShadAlert.destructive(
              icon: const Icon(LucideIcons.circleAlert, size: 16),
              title: Text(_mismatch),
              description: const Text(
                'Doğru dersin sayfasından ekleyin veya doğru Word dosyasını seçin.',
              ),
            ),
            const SizedBox(height: 8),
          ],
          for (final warning in info.warnings) ...[
            ShadAlert(
              icon: const Icon(LucideIcons.triangleAlert, size: 16),
              title: Text(warning),
            ),
            const SizedBox(height: 8),
          ],
          const SizedBox(height: 8),
          Text('Sorular', style: theme.textTheme.h4),
          const SizedBox(height: 8),
          for (final question in info.questions)
            _QuestionRow(
              question: question,
              onMaximum: (value) =>
                  setState(() => _maximums[question.id] = value),
            ),
          const SizedBox(height: 16),
          Text('Öğrenci listesi', style: theme.textTheme.h4),
          const SizedBox(height: 4),
          Text(
            'İsteğe bağlı. Her satırda “numara;ad soyad”. Öğrenci adı kâğıttan okunmaz, numaraya göre listeden bulunur.',
            style: theme.textTheme.muted,
          ),
          const SizedBox(height: 8),
          ShadButton.outline(
            leading: const Icon(LucideIcons.users, size: 16),
            onPressed: _pickRoster,
            child: Text(
              _rosterFile == null
                  ? 'Liste seç (CSV)'
                  : '$_rosterFile · ${_roster.length} öğrenci',
            ),
          ),
          const SizedBox(height: 24),
          ShadButton(
            width: double.infinity,
            enabled: _mismatch == null && _complete && !_busy,
            onPressed: _create,
            child: Text(_busy ? 'Hazırlanıyor…' : 'Sınavı oluştur'),
          ),
          if (_mismatch == null && !_complete)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Devam etmek için eksik tam puanları girin.',
                style: theme.textTheme.muted,
                textAlign: TextAlign.center,
              ),
            ),
        ],
      ),
    );
  }
}

class _QuestionRow extends StatelessWidget {
  const _QuestionRow({required this.question, required this.onMaximum});

  final QuestionInfo question;
  final ValueChanged<String> onMaximum;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 44,
            child: Text(question.id, style: theme.textTheme.large),
          ),
          Expanded(
            child: Wrap(
              spacing: 6,
              children: [
                for (final outcome in question.outcomes)
                  ShadBadge.secondary(child: Text(outcome)),
              ],
            ),
          ),
          if (question.maximumScore != null)
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('${question.maximumScore} puan', style: theme.textTheme.p),
                if (question.maximumSource != null)
                  Text(question.maximumSource!, style: theme.textTheme.muted),
              ],
            )
          else
            SizedBox(
              width: 110,
              child: ShadInput(
                placeholder: const Text('Tam puan'),
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
                ],
                onChanged: onMaximum,
              ),
            ),
        ],
      ),
    );
  }
}
