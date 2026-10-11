import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../review_labels.dart';
import '../src/rust/api/reader.dart';
import '../storage.dart';

/// Every recognised cell next to the photo crop it came from. The instructor
/// checks or corrects the values and confirms the sheet.
class SheetReviewPage extends StatefulWidget {
  const SheetReviewPage({
    super.key,
    required this.exam,
    required this.record,
    this.fresh = false,
  });

  final ExamEntry exam;
  final SheetRecord record;

  /// True for a sheet that was just scanned and is not saved yet.
  final bool fresh;

  @override
  State<SheetReviewPage> createState() => _SheetReviewPageState();
}

class _SheetReviewPageState extends State<SheetReviewPage> {
  static const _confident = 0.8;

  late SheetResult _result = widget.record.result;
  late final Map<String, TextEditingController> _values = {
    for (final cell in widget.record.result.cells)
      cell.field: TextEditingController(text: cell.value ?? '')
        ..addListener(_edited),
  };

  /// A confirmation that still has warnings; saving it needs a second tap.
  SheetResult? _confirmedWithWarnings;
  bool _busy = false;

  void _edited() {
    if (_confirmedWithWarnings != null) {
      setState(() => _confirmedWithWarnings = null);
    }
  }

  @override
  void dispose() {
    for (final controller in _values.values) {
      controller.dispose();
    }
    super.dispose();
  }

  /// Keeps the photo crops of the original reading.
  SheetResult _withCrops(SheetResult result) {
    final crops = {
      for (final c in widget.record.result.cells) c.field: c.cropPng,
    };
    return SheetResult(
      sheetId: result.sheetId,
      pageFound: widget.record.result.pageFound,
      studentNumber: result.studentNumber,
      studentName: result.studentName,
      cells: [
        for (final c in result.cells)
          CellResult(
            field: c.field,
            text: c.text,
            value: c.value,
            confidence: c.confidence,
            accepted: c.accepted,
            region: c.region,
            cropPng: crops[c.field] ?? c.cropPng,
          ),
      ],
      issues: result.issues,
      outcomes: result.outcomes,
      requiresReview: result.requiresReview,
    );
  }

  Future<void> _confirm() async {
    setState(() => _busy = true);
    try {
      var confirmed = _confirmedWithWarnings;
      if (confirmed == null) {
        final reader = await widget.exam.reader();
        confirmed = _withCrops(
          await reader.confirmSheet(
            sheetId: widget.record.id,
            values: [
              for (final cell in widget.record.result.cells)
                FieldValue(
                  field: cell.field,
                  value: _values[cell.field]!.text.trim().isEmpty
                      ? null
                      : _values[cell.field]!.text.trim(),
                  region: cell.region,
                ),
            ],
          ),
        );
        if (confirmed.issues.isNotEmpty) {
          setState(() {
            _result = confirmed!;
            _confirmedWithWarnings = confirmed;
          });
          return;
        }
      }
      await _save(confirmed, confirmed: true);
    } catch (error) {
      _toast('Onaylanamadı', '$error', destructive: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save(SheetResult result, {required bool confirmed}) async {
    await appState.saveSheet(
      widget.exam,
      SheetRecord(
        id: widget.record.id,
        createdAt: widget.record.createdAt,
        confirmed: confirmed,
        result: result,
      ),
    );
    if (confirmed && !widget.record.confirmed) {
      // What the model read versus what the instructor confirmed.
      await appState.addCorrections([
        for (final cell in widget.record.result.cells)
          if (cell.cropPng.isNotEmpty &&
              (cell.value ?? '') != _values[cell.field]!.text.trim())
            Correction(
              examId: widget.exam.id,
              sheetId: widget.record.id,
              field: cell.field,
              recognized: cell.text,
              corrected: _values[cell.field]!.text.trim().isEmpty
                  ? null
                  : _values[cell.field]!.text.trim(),
              confidence: cell.confidence,
              crop: cell.cropPng,
            ),
      ]);
    }
    if (!mounted) return;
    _toast(confirmed ? 'Kâğıt onaylandı' : 'Kâğıt kaydedildi', null);
    Navigator.of(context).pop();
  }

  Future<void> _delete() async {
    final yes = await showShadDialog<bool>(
      context: context,
      builder: (context) => ShadDialog.alert(
        title: const Text('Kâğıt silinsin mi?'),
        description: const Text(
          'Bu kâğıdın okunan değerleri ve fotoğraf parçaları silinir.',
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
    await appState.deleteSheet(widget.exam, widget.record);
    if (mounted) Navigator.of(context).pop();
  }

  void _toast(String title, String? description, {bool destructive = false}) {
    if (!mounted) return;
    final toast = destructive
        ? ShadToast.destructive(
            title: Text(title),
            description: description == null ? null : Text(description),
          )
        : ShadToast(
            title: Text(title),
            description: description == null ? null : Text(description),
          );
    ShadToaster.of(context).show(toast);
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final verified = widget.record.confirmed;
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text(
          widget.fresh ? 'Okunan kâğıt' : 'Kâğıt',
          style: theme.textTheme.large,
        ),
        actions: [
          if (!widget.fresh)
            ShadIconButton.ghost(
              icon: const Icon(LucideIcons.trash2, size: 18),
              onPressed: _delete,
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ShadCard(
            title: Text(_result.studentName ?? 'Öğrenci belirlenemedi'),
            description: Text(
              _result.studentNumber ?? 'Numara listede bulunamadı',
            ),
            trailing: verified && _confirmedWithWarnings == null
                ? const ShadBadge(child: Text('Onaylandı'))
                : const ShadBadge.outline(child: Text('Kontrol bekliyor')),
          ),
          const SizedBox(height: 12),
          if (!_result.pageFound) ...[
            const ShadAlert.destructive(
              icon: Icon(LucideIcons.circleAlert, size: 16),
              title: Text('Kâğıt formu net tanınamadı'),
              description: Text(
                'Başlık ve Soru–PÇ tablosu görünecek şekilde yeniden çekmeniz önerilir.',
              ),
            ),
            const SizedBox(height: 12),
          ],
          for (final issue in _result.issues) ...[
            ShadAlert(
              icon: const Icon(LucideIcons.info, size: 16),
              title: Text(reviewText(issue)),
              description: issue.detail == null ? null : Text(issue.detail!),
            ),
            const SizedBox(height: 8),
          ],
          const SizedBox(height: 8),
          Text('Değerler', style: theme.textTheme.h4),
          const SizedBox(height: 4),
          Text(
            'Fotoğraftaki hücreyle karşılaştırın; yanlışsa düzeltin, boşsa boş bırakın.',
            style: theme.textTheme.muted,
          ),
          const SizedBox(height: 8),
          for (final cell in widget.record.result.cells)
            _CellRow(
              cell: cell,
              controller: _values[cell.field]!,
              showModel: !verified,
              confident: cell.confidence >= _confident,
            ),
          const SizedBox(height: 16),
          if (_result.outcomes.isNotEmpty) ...[
            Text('PÇ sonuçları', style: theme.textTheme.h4),
            const SizedBox(height: 8),
            for (final outcome in _result.outcomes)
              _OutcomeRow(outcome: outcome),
          ],
          const SizedBox(height: 24),
          ShadButton(
            width: double.infinity,
            enabled: !_busy,
            leading: const Icon(LucideIcons.circleCheck, size: 16),
            onPressed: _confirm,
            child: Text(
              _confirmedWithWarnings != null
                  ? 'Uyarılara rağmen onayla'
                  : 'Onayla ve kaydet',
            ),
          ),
          if (widget.fresh) ...[
            const SizedBox(height: 8),
            ShadButton.outline(
              width: double.infinity,
              enabled: !_busy,
              onPressed: () => _save(_result, confirmed: false),
              child: const Text('Sonra kontrol et'),
            ),
          ],
        ],
      ),
    );
  }
}

class _CellRow extends StatelessWidget {
  const _CellRow({
    required this.cell,
    required this.controller,
    required this.showModel,
    required this.confident,
  });

  final CellResult cell;
  final TextEditingController controller;
  final bool showModel;
  final bool confident;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final isNumber = cell.field == 'Numara';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 56,
            child: Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(cell.field, style: theme.textTheme.small),
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Container(
                    height: 44,
                    alignment: Alignment.centerLeft,
                    decoration: BoxDecoration(
                      border: Border.all(color: theme.colorScheme.border),
                    ),
                    child: cell.cropPng.isEmpty
                        ? Text('  fotoğraf yok', style: theme.textTheme.muted)
                        : Image.memory(
                            cell.cropPng,
                            fit: BoxFit.contain,
                            gaplessPlayback: true,
                          ),
                  ),
                ),
                if (showModel)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      'Okunan: ${cell.text.isEmpty ? 'boş' : cell.text} · güven %${(cell.confidence * 100).round()}',
                      style: confident
                          ? theme.textTheme.muted
                          : theme.textTheme.muted.copyWith(
                              color: theme.colorScheme.destructive,
                            ),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            width: isNumber ? 130 : 76,
            child: ShadInput(
              controller: controller,
              keyboardType: isNumber
                  ? TextInputType.number
                  : const TextInputType.numberWithOptions(decimal: true),
              inputFormatters: [
                FilteringTextInputFormatter.allow(
                  RegExp(isNumber ? r'[0-9]' : r'[0-9.,]'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _OutcomeRow extends StatelessWidget {
  const _OutcomeRow({required this.outcome});

  final OutcomeResult outcome;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 56,
            child: Text(outcome.id, style: theme.textTheme.large),
          ),
          Expanded(child: ShadProgress(value: outcome.percent / 100)),
          SizedBox(
            width: 96,
            child: outcome.complete
                ? Text(
                    '${outcome.earned}/${outcome.possible}',
                    textAlign: TextAlign.end,
                    style: theme.textTheme.p,
                  )
                : const Align(
                    alignment: Alignment.centerRight,
                    child: ShadBadge.outline(child: Text('eksik')),
                  ),
          ),
        ],
      ),
    );
  }
}
