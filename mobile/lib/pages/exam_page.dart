import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:share_plus/share_plus.dart';

import '../app_state.dart';
import '../src/rust/api/reader.dart';
import '../storage.dart';
import 'sheet_review_page.dart';

class ExamPage extends StatefulWidget {
  const ExamPage({super.key, required this.exam});

  final ExamEntry exam;

  @override
  State<ExamPage> createState() => _ExamPageState();
}

class _ExamPageState extends State<ExamPage> {
  bool _reading = false;
  bool _exporting = false;

  Future<void> _scan(ImageSource source) async {
    final photo = await ImagePicker().pickImage(
      source: source,
      maxWidth: 2400,
      maxHeight: 2400,
      imageQuality: 92,
    );
    if (photo == null || !mounted) return;
    await _read(await photo.readAsBytes());
  }

  Future<void> _read(Uint8List photo) async {
    if (!mounted) return;
    setState(() => _reading = true);
    try {
      final reader = await widget.exam.reader();
      final result = await reader.processPhoto(
        sheetId: DateTime.now().microsecondsSinceEpoch.toString(),
        photo: photo,
      );
      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => SheetReviewPage(
            exam: widget.exam,
            fresh: true,
            record: SheetRecord(
              id: result.sheetId,
              createdAt: DateTime.now(),
              confirmed: false,
              result: result,
            ),
          ),
        ),
      );
    } catch (error) {
      _toast('Kâğıt okunamadı', '$error');
    } finally {
      if (mounted) setState(() => _reading = false);
    }
  }

  Future<void> _export() async {
    final exam = widget.exam;
    if (exam.sheets.isEmpty) {
      _toast('Aktarılacak kâğıt yok', 'Önce kâğıt tarayın.');
      return;
    }
    setState(() => _exporting = true);
    try {
      final reader = await exam.reader();
      final bytes = await reader.exportExcel(
        sheets: [
          for (final record in exam.sheets.reversed)
            ReportSheet(
              studentNumber: record.result.studentNumber,
              studentName: record.result.studentName,
              values: [
                for (final cell in record.result.cells)
                  FieldValue(
                    field: cell.field,
                    // Unconfirmed sheets report only values the app accepted.
                    value: record.confirmed || cell.accepted
                        ? cell.value
                        : null,
                    region: cell.region,
                  ),
              ],
              outcomes: record.result.outcomes,
              confirmed: record.confirmed,
            ),
        ],
      );
      final name = [
        exam.info.courseCode ?? 'sinav',
        exam.info.examKind,
        if (exam.info.group != null) 'grup-${exam.info.group}',
      ].whereType<String>().join('-');
      final file = File('${(await getTemporaryDirectory()).path}/$name.xlsx');
      await file.writeAsBytes(bytes, flush: true);
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(file.path)],
          subject: '${exam.title} – PÇ raporu',
        ),
      );
    } catch (error) {
      _toast('Excel oluşturulamadı', '$error');
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  Future<void> _delete() async {
    final exam = widget.exam;
    final yes = await showShadDialog<bool>(
      context: context,
      builder: (context) => ShadDialog.alert(
        title: Text('${exam.label} silinsin mi?'),
        description: Text(
          'Sınavın ${exam.sheets.length} kâğıdı ve fotoğraf parçaları da silinir.',
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
    await appState.deleteExam(exam);
    if (mounted) Navigator.of(context).pop();
  }

  void _toast(String title, String description) {
    if (!mounted) return;
    ShadToaster.of(context).show(
      ShadToast.destructive(title: Text(title), description: Text(description)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final exam = widget.exam;
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.background,
        surfaceTintColor: Colors.transparent,
        title: Text(
          '${exam.info.courseCode ?? exam.title} · ${exam.label}',
          style: theme.textTheme.large,
        ),
        actions: [
          ShadIconButton.ghost(
            icon: const Icon(LucideIcons.trash2, size: 18),
            onPressed: _delete,
          ),
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: ShadButton.outline(
              size: ShadButtonSize.sm,
              enabled: !_exporting,
              leading: const Icon(LucideIcons.fileSpreadsheet, size: 16),
              onPressed: _export,
              child: Text(_exporting ? 'Hazırlanıyor…' : 'Excel'),
            ),
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: appState,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(exam.subtitle, style: theme.textTheme.muted),
            const SizedBox(height: 16),
            ShadTabs<String>(
              value: 'sheets',
              tabs: [
                ShadTab(
                  value: 'sheets',
                  content: _SheetsTab(
                    exam: exam,
                    reading: _reading,
                    onScan: _scan,
                  ),
                  child: const Text('Kâğıtlar'),
                ),
                ShadTab(
                  value: 'results',
                  content: _ResultsTab(exam: exam),
                  child: const Text('PÇ sonuçları'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SheetsTab extends StatelessWidget {
  const _SheetsTab({
    required this.exam,
    required this.reading,
    required this.onScan,
  });

  final ExamEntry exam;
  final bool reading;
  final void Function(ImageSource) onScan;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final waiting = exam.sheets.where((s) => !s.confirmed).length;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: ShadButton(
                enabled: !reading,
                leading: const Icon(LucideIcons.camera, size: 16),
                onPressed: () => onScan(ImageSource.camera),
                child: const Text('Kâğıt tara'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: ShadButton.outline(
                enabled: !reading,
                leading: const Icon(LucideIcons.image, size: 16),
                onPressed: () => onScan(ImageSource.gallery),
                child: const Text('Galeriden'),
              ),
            ),
          ],
        ),
        if (reading) ...[
          const SizedBox(height: 8),
          const ShadProgress(),
          const SizedBox(height: 6),
          Text(
            'Kâğıt okunuyor…',
            style: theme.textTheme.muted,
            textAlign: TextAlign.center,
          ),
        ],
        const SizedBox(height: 12),
        if (exam.sheets.isEmpty)
          ShadCard(
            title: const Text('Henüz kâğıt yok'),
            description: const Text(
              'Kâğıdın üst kısmı (başlık ve Soru–PÇ tablosu) kadrajda olacak şekilde fotoğraf çekin.',
            ),
          )
        else ...[
          Text(
            '${exam.sheets.length} kâğıt'
            '${waiting > 0 ? ' · $waiting tanesi kontrol bekliyor' : ' · hepsi onaylı'}',
            style: theme.textTheme.muted,
          ),
          const SizedBox(height: 8),
          for (final record in exam.sheets) ...[
            _SheetTile(exam: exam, record: record),
            const SizedBox(height: 10),
          ],
        ],
      ],
    );
  }
}

class _SheetTile extends StatelessWidget {
  const _SheetTile({required this.exam, required this.record});

  final ExamEntry exam;
  final SheetRecord record;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final sheet = record.result;
    final scores = sheet.cells
        .where((c) => c.field != 'Numara')
        .map((c) => '${c.field}: ${c.value ?? '–'}');
    return GestureDetector(
      onTap: () => Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => SheetReviewPage(exam: exam, record: record),
        ),
      ),
      child: ShadCard(
        title: Text(sheet.studentName ?? 'Öğrenci belirlenemedi'),
        description: Text(sheet.studentNumber ?? 'Numara kontrol edilmeli'),
        trailing: record.confirmed
            ? const ShadBadge(child: Text('Onaylandı'))
            : const ShadBadge.outline(child: Text('Kontrol bekliyor')),
        child: Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(scores.join('   '), style: theme.textTheme.small),
        ),
      ),
    );
  }
}

class _ResultsTab extends StatelessWidget {
  const _ResultsTab({required this.exam});

  final ExamEntry exam;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    // Class result per PÇ from the sheets whose PÇ scores are final.
    final earned = <String, double>{};
    final possible = <String, double>{};
    final counted = <String, int>{};
    for (final record in exam.sheets) {
      for (final outcome in record.result.outcomes) {
        possible.putIfAbsent(outcome.id, () => 0);
        earned.putIfAbsent(outcome.id, () => 0);
        if (!outcome.complete) continue;
        earned[outcome.id] = earned[outcome.id]! + double.parse(outcome.earned);
        possible[outcome.id] =
            possible[outcome.id]! + double.parse(outcome.possible);
        counted[outcome.id] = (counted[outcome.id] ?? 0) + 1;
      }
    }
    final ids = possible.keys.toList()..sort();
    if (ids.isEmpty) {
      return Padding(
        padding: const EdgeInsets.only(top: 16),
        child: ShadCard(
          title: const Text('Sonuç yok'),
          description: const Text(
            'Kâğıt taradıkça PÇ başarı oranları burada görünür.',
          ),
        ),
      );
    }
    final confirmed = exam.sheets.where((s) => s.confirmed).length;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 12),
        Text(
          '${exam.sheets.length} kâğıt, $confirmed onaylı. Puanı kesinleşmemiş PÇ\'ler o kâğıt için sayılmaz.',
          style: theme.textTheme.muted,
        ),
        const SizedBox(height: 12),
        for (final id in ids) ...[
          Row(
            children: [
              SizedBox(
                width: 56,
                child: Text(id, style: theme.textTheme.large),
              ),
              Expanded(
                child: ShadProgress(
                  value: possible[id]! == 0 ? 0 : earned[id]! / possible[id]!,
                ),
              ),
              SizedBox(
                width: 64,
                child: Text(
                  possible[id]! == 0
                      ? '–'
                      : '%${(100 * earned[id]! / possible[id]!).toStringAsFixed(1)}',
                  textAlign: TextAlign.end,
                  style: theme.textTheme.p,
                ),
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.only(left: 56, top: 2, bottom: 12),
            child: Text(
              '${counted[id] ?? 0} kâğıttan',
              style: theme.textTheme.muted,
            ),
          ),
        ],
      ],
    );
  }
}
