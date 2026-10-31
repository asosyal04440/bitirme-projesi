import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'app_state.dart';
import 'src/rust/api/reader.dart';

/// Test recordings: while one runs, the scanning screen keeps every camera
/// frame (see `recording.rs` in the Rust library). Shared together with the
/// sheets' checked values, they are a test set of real sheets: the frames are
/// replayed on a computer and what is read is compared with the values.

Future<Directory> _recordings(ExamEntry exam) async => Directory(
  '${(await getApplicationDocumentsDirectory()).path}/kayitlar/${exam.id}',
);

/// Starts keeping the frames the exam's reader scans; returns the folder.
Future<String> startTestRecording(ExamEntry exam) async {
  final stamp = DateTime.now().toIso8601String().replaceAll(
    RegExp(r'[:.]'),
    '-',
  );
  final dir = '${(await _recordings(exam)).path}/$stamp';
  await (await exam.reader()).startRecording(dir: dir);
  return dir;
}

Future<void> stopTestRecording(ExamEntry exam) async =>
    (await exam.reader()).stopRecording();

/// Whether the exam has test recordings to share.
Future<bool> hasTestRecordings(ExamEntry exam) async {
  final root = await _recordings(exam);
  return root.existsSync() && root.listSync().whereType<Directory>().isNotEmpty;
}

/// Packs every recording of the exam with the sheets as they are now (values
/// read or corrected, and whether the instructor confirmed them) and the
/// exam's definition, and shares the zips. Returns how many there were.
Future<int> shareTestRecordings(ExamEntry exam) async {
  final root = await _recordings(exam);
  if (!root.existsSync()) return 0;
  final sheets = [
    for (final record in exam.sheets)
      {
        'sheet': record.id,
        'confirmed': record.confirmed,
        'values': {
          for (final cell in record.result.cells) cell.field: cell.value,
        },
      },
  ];
  final extra = [
    RecordingFile(
      name: 'labels.json',
      content: utf8.encode(jsonEncode(sheets)),
    ),
    if (exam.stored.definition != null)
      RecordingFile(
        name: 'definition.json',
        content: utf8.encode(jsonEncode(exam.stored.definition)),
      ),
  ];
  final zips = <XFile>[];
  for (final dir in root.listSync().whereType<Directory>()) {
    zips.add(XFile(await packRecording(dir: dir.path, extra: extra)));
  }
  if (zips.isEmpty) return 0;
  await SharePlus.instance.share(
    ShareParams(files: zips, subject: 'Sınav Okuyucu test kaydı'),
  );
  return zips.length;
}
