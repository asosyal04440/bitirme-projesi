import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

import '../app_state.dart';
import '../course_students.dart';
import 'sheet_review_page.dart';

/// Every student of the course with their PÇ in each exam and overall.
class StudentsTab extends StatefulWidget {
  const StudentsTab({super.key, required this.course});

  final CourseEntry course;

  @override
  State<StudentsTab> createState() => _StudentsTabState();
}

class _StudentsTabState extends State<StudentsTab> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final (:students, :unidentified) = courseStudents(widget.course);
    final query = _query.trim().toLowerCase();
    final shown = query.isEmpty
        ? students
        : students
              .where(
                (s) =>
                    s.number.contains(query) ||
                    (s.name?.toLowerCase().contains(query) ?? false),
              )
              .toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 12),
        if (students.isEmpty && unidentified.isEmpty)
          ShadCard(
            title: const Text('Henüz öğrenci yok'),
            description: const Text(
              'Sınavlardan kâğıt okuttukça öğrenciler numaralarına göre burada toplanır.',
            ),
          )
        else ...[
          ShadInput(
            leading: const Icon(LucideIcons.search, size: 16),
            placeholder: const Text('Numara veya ad ara'),
            onChanged: (value) => setState(() => _query = value),
          ),
          const SizedBox(height: 8),
          Text(
            '${students.length} öğrenci. Ders PÇ’si: vize %40 + final %60; '
            'bütünleme finalin, mazeret vizenin yerine geçer.',
            style: theme.textTheme.muted,
          ),
          const SizedBox(height: 12),
          for (final student in shown) ...[
            _StudentCard(student: student),
            const SizedBox(height: 12),
          ],
          if (unidentified.isNotEmpty)
            ShadAlert(
              icon: const Icon(LucideIcons.info, size: 16),
              title: Text(
                '${unidentified.length} kâğıdın öğrenci numarası okunamadı',
              ),
              description: const Text(
                'Kâğıdı açıp numarayı düzeltince öğrenci burada görünür.',
              ),
            ),
        ],
      ],
    );
  }
}

class _StudentCard extends StatelessWidget {
  const _StudentCard({required this.student});

  final StudentSummary student;

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    final outcomes = {
      for (final exam in student.exams)
        for (final o in exam.outcomes) o.id,
    }.toList()..sort(compareOutcomeIds);
    final counted = student.counted.toSet();
    final totals = {for (final t in student.totals) t.id: t};

    Widget cell(String text, {TextStyle? style, Alignment? align}) => Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 7),
      child: Align(
        alignment: align ?? Alignment.centerRight,
        child: Text(text, style: style ?? theme.textTheme.small),
      ),
    );

    // "—" when the exam has no question for the PÇ, "eksik" while a score
    // of it is not final.
    Widget value(double? percent, {required bool complete, bool bold = false}) {
      if (percent == null) return cell('—', style: theme.textTheme.muted);
      if (!complete) return cell('eksik', style: theme.textTheme.muted);
      return cell(
        '%${percent.toStringAsFixed(0)}',
        style: bold
            ? theme.textTheme.small.copyWith(fontWeight: FontWeight.w600)
            : null,
      );
    }

    Widget outcomeOf(StudentExam exam, String id) {
      final o = exam.outcomes.where((o) => o.id == id).firstOrNull;
      // Points until the exam's maximum scores are entered.
      if (o != null && o.complete && !o.maximumKnown) {
        return cell('${o.earned} p', style: theme.textTheme.muted);
      }
      return value(o?.percent, complete: o?.complete ?? true);
    }

    return ShadCard(
      title: Text(student.name ?? student.number),
      description: student.name == null ? null : Text(student.number),
      trailing: student.allConfirmed
          ? null
          : const ShadBadge.outline(child: Text('Kontrol bekliyor')),
      child: Padding(
        padding: const EdgeInsets.only(top: 12),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Table(
            defaultColumnWidth: const IntrinsicColumnWidth(),
            defaultVerticalAlignment: TableCellVerticalAlignment.middle,
            children: [
              TableRow(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: theme.colorScheme.border),
                  ),
                ),
                children: [
                  cell('', align: Alignment.centerLeft),
                  for (final id in outcomes)
                    cell(
                      id,
                      style: theme.textTheme.small.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                ],
              ),
              for (final exam in student.exams)
                TableRow(
                  children: [
                    GestureDetector(
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => SheetReviewPage(
                            exam: exam.exam,
                            record: exam.sheet,
                          ),
                        ),
                      ),
                      child: cell(
                        counted.contains(exam)
                            ? exam.kind
                            : '${exam.kind} (sayılmaz)',
                        align: Alignment.centerLeft,
                        style: theme.textTheme.small.copyWith(
                          color: counted.contains(exam)
                              ? null
                              : theme.colorScheme.mutedForeground,
                          decoration: TextDecoration.underline,
                          decorationColor: theme.colorScheme.border,
                        ),
                      ),
                    ),
                    for (final id in outcomes) outcomeOf(exam, id),
                  ],
                ),
              TableRow(
                decoration: BoxDecoration(
                  color: theme.colorScheme.muted,
                  borderRadius: BorderRadius.circular(6),
                ),
                children: [
                  cell(
                    'Ders',
                    align: Alignment.centerLeft,
                    style: theme.textTheme.small.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  for (final id in outcomes)
                    value(
                      totals[id]?.percent,
                      complete: totals[id]?.complete ?? true,
                      bold: true,
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// "PÇ2" before "PÇ10".
int compareOutcomeIds(String a, String b) {
  int number(String id) =>
      int.tryParse(RegExp(r'\d+').firstMatch(id)?.group(0) ?? '') ?? 0;
  final byNumber = number(a).compareTo(number(b));
  return byNumber != 0 ? byNumber : a.compareTo(b);
}
