import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../definition_json.dart';

/// "PÇ2", "pc 2", "2" → "PÇ2"; several separated by commas or spaces.
List<String> parseOutcomes(String text) => [
  for (final match in RegExp(r'\d+').allMatches(text)) 'PÇ${match.group(0)}',
];

/// What was read off the sheet of an exam recognised there, to correct, and
/// what the sheet does not say (maximum scores) to fill in. None of it holds
/// up scanning: until the maximum scores are given, PÇ are shown in points.
/// Pops with the updated exam.
class ExamDetailsPage extends StatefulWidget {
  const ExamDetailsPage({super.key, required this.exam});

  final ExamEntry exam;

  @override
  State<ExamDetailsPage> createState() => _ExamDetailsPageState();
}

class _ExamDetailsPageState extends State<ExamDetailsPage> {
  late final _definition = definitionFromJson(widget.exam.stored.definition!);
  late final _code = TextEditingController(text: _definition.courseCode ?? '');
  late final _group = TextEditingController(text: _definition.group ?? '');
  late String? _kind = _definition.examKind;
  late final Map<String, TextEditingController> _outcomes = {
    for (final q in _definition.questions)
      q.printedId: TextEditingController(text: q.outcomes.join(', ')),
  };
  late final Map<String, TextEditingController> _maximums = {
    for (final q in _definition.questions)
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

  static String _number(String text) => text.trim().replaceAll(',', '.');

  /// Entered maximum scores must be positive numbers; blank ones may wait.
  bool get _valid => _maximums.values.every((c) {
    final text = _number(c.text);
    return text.isEmpty || (double.tryParse(text) ?? 0) > 0;
  });

  Future<void> _save() async {
    setState(() => _busy = true);
    try {
      final code = normalizeCourseCode(_code.text);
      final group = _group.text.trim().toUpperCase();
      final updated = await appState.updateDetectedExam(
        widget.exam,
        definitionWith(
          _definition,
          courseCode: code.isEmpty ? null : code,
          examKind: _kind,
          group: group.isEmpty ? null : group,
          outcomes: {
            for (final e in _outcomes.entries)
              if (parseOutcomes(e.value.text).isNotEmpty)
                e.key: parseOutcomes(e.value.text),
          },
          maximums: {
            for (final e in _maximums.entries)
              if (_number(e.value.text).isNotEmpty)
                e.key: _number(e.value.text),
          },
        ),
      );
      if (mounted) Navigator.of(context).pop(updated);
    } catch (error) {
      if (!mounted) return;
      ShadToaster.of(context).show(
        ShadToast.destructive(
          title: const Text('Kaydedilemedi'),
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
    final d = _definition;
    final details = [
      if (d.group != null) 'Grup ${d.group}',
      [d.academicYear, d.term].whereType<String>().join(' '),
    ].where((s) => s.isNotEmpty).join(' · ');
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text('Sınav bilgileri', style: theme.textTheme.large),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ShadCard(
            title: Text(d.courseName ?? d.course ?? 'Ders'),
            description: details.isEmpty ? null : Text(details),
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
            'Soru–PÇ eşleşmeleri kâğıttan okundu. Tam puanlar kâğıtta '
            'yazmıyor: girilene kadar PÇ’ler puan olarak gösterilir, '
            'girilince yüzdeler hesaplanır.',
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
            enabled: _valid && !_busy,
            onPressed: _save,
            child: Text(_busy ? 'Kaydediliyor…' : 'Kaydet'),
          ),
        ],
      ),
    );
  }
}
