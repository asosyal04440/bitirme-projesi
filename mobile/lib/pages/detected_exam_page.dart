import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../definition_json.dart';
import '../src/rust/api/definition.dart';
import 'live_scan_page.dart';

/// "PÇ2", "pc 2", "2" → "PÇ2"; several separated by commas or spaces.
List<String> parseOutcomes(String text) => [
  for (final match in RegExp(r'\d+').allMatches(text)) 'PÇ${match.group(0)}',
];

/// An exam recognised on its first sheet: what was read is shown for a
/// check, and only what the sheet does not say (maximum scores) is asked.
class DetectedExamPage extends StatefulWidget {
  const DetectedExamPage({super.key, required this.definition});

  final ExamDefinition definition;

  @override
  State<DetectedExamPage> createState() => _DetectedExamPageState();
}

class _DetectedExamPageState extends State<DetectedExamPage> {
  late final _code = TextEditingController(
    text: widget.definition.courseCode ?? '',
  );
  late final _group = TextEditingController(
    text: widget.definition.group ?? '',
  );
  late String? _kind = widget.definition.examKind;
  late final Map<String, TextEditingController> _outcomes = {
    for (final q in widget.definition.questions)
      q.printedId: TextEditingController(text: q.outcomes.join(', '))
        ..addListener(_changed),
  };
  late final Map<String, TextEditingController> _maximums = {
    for (final q in widget.definition.questions)
      q.printedId: TextEditingController(text: q.maximum ?? '')
        ..addListener(_changed),
  };
  bool _busy = false;

  void _changed() => setState(() {});

  @override
  void dispose() {
    _code.dispose();
    _group.dispose();
    for (final c in [..._outcomes.values, ..._maximums.values]) {
      c.dispose();
    }
    super.dispose();
  }

  bool get _complete =>
      normalizeCourseCode(_code.text).isNotEmpty &&
      _kind != null &&
      _outcomes.values.every((c) => parseOutcomes(c.text).isNotEmpty) &&
      _maximums.values.every((c) => c.text.trim().isNotEmpty);

  Future<void> _start() async {
    setState(() => _busy = true);
    try {
      final exam = await appState.createDetectedExam(
        definitionWith(
          widget.definition,
          courseCode: normalizeCourseCode(_code.text),
          examKind: _kind,
          group: _group.text.trim().isEmpty
              ? null
              : _group.text.trim().toUpperCase(),
          outcomes: {
            for (final e in _outcomes.entries)
              e.key: parseOutcomes(e.value.text),
          },
          maximums: {
            for (final e in _maximums.entries) e.key: e.value.text.trim(),
          },
        ),
      );
      if (!mounted) return;
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(builder: (_) => LiveScanPage(exam: exam)),
      );
    } catch (error) {
      if (!mounted) return;
      ShadToaster.of(context).show(
        ShadToast.destructive(
          title: const Text('Sınav kaydedilemedi'),
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
    final d = widget.definition;
    final details = [
      d.examKind,
      if (d.group != null) 'Grup ${d.group}',
      [d.academicYear, d.term].whereType<String>().join(' '),
    ].whereType<String>().where((s) => s.isNotEmpty).join(' · ');
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text('Sınav tanındı', style: theme.textTheme.large),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ShadCard(
            title: Text(d.courseName ?? d.course ?? 'Ders'),
            description: Text(
              details.isEmpty ? 'Sınav türü okunamadı' : details,
            ),
            child: Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Row(
                children: [
                  Text('Ders kodu', style: theme.textTheme.small),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ShadInput(
                      controller: _code,
                      textCapitalization: TextCapitalization.characters,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text('Sınav türü', style: theme.textTheme.small),
          const SizedBox(height: 6),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final kind in const [
                'Vize',
                'Final',
                'Bütünleme',
                'Mazeret',
              ])
                kind == _kind
                    ? ShadButton(
                        size: ShadButtonSize.sm,
                        onPressed: () {},
                        child: Text(kind),
                      )
                    : ShadButton.outline(
                        size: ShadButtonSize.sm,
                        onPressed: () => setState(() => _kind = kind),
                        child: Text(kind),
                      ),
              SizedBox(
                width: 110,
                child: ShadInput(
                  controller: _group,
                  placeholder: const Text('Grup'),
                  textCapitalization: TextCapitalization.characters,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            'Kâğıttan okunan soru–PÇ eşleşmelerini kontrol edin ve her sorunun '
            'tam puanını girin (tam puanlar kâğıtta yazmıyor).',
            style: theme.textTheme.muted,
          ),
          const SizedBox(height: 12),
          for (final q in d.questions)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Row(
                children: [
                  SizedBox(
                    width: 44,
                    child: Text(q.printedId, style: theme.textTheme.large),
                  ),
                  Expanded(
                    child: ShadInput(
                      controller: _outcomes[q.printedId],
                      placeholder: const Text('PÇ'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 110,
                    child: ShadInput(
                      controller: _maximums[q.printedId],
                      placeholder: const Text('Tam puan'),
                      keyboardType: const TextInputType.numberWithOptions(
                        decimal: true,
                      ),
                      inputFormatters: [
                        FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          const SizedBox(height: 24),
          ShadButton(
            width: double.infinity,
            enabled: _complete && !_busy,
            leading: const Icon(LucideIcons.scanLine, size: 16),
            onPressed: _start,
            child: Text(_busy ? 'Hazırlanıyor…' : 'Taramaya başla'),
          ),
          if (!_complete)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Devam etmek için sınav türünü, her sorunun PÇ’sini ve tam '
                'puanını girin.',
                style: theme.textTheme.muted,
                textAlign: TextAlign.center,
              ),
            ),
        ],
      ),
    );
  }
}
