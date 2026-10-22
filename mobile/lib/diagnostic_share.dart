import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'app_state.dart';

/// Shares what the reader saw in the last photo or camera frame: the picture
/// with the located page, table and cells drawn on it, and a text report.
/// Returns false when nothing has been read yet.
Future<bool> shareDiagnostic(ExamEntry exam) async {
  final diagnostic = await (await exam.reader()).lastDiagnostic();
  if (diagnostic == null) return false;
  final file = File(
    '${(await getTemporaryDirectory()).path}/'
    'tani-${DateTime.now().millisecondsSinceEpoch}.png',
  );
  await file.writeAsBytes(diagnostic.imagePng, flush: true);
  await SharePlus.instance.share(
    ShareParams(
      files: [XFile(file.path)],
      text: diagnostic.report,
      subject: 'Sınav Okuyucu tanı bilgisi',
    ),
  );
  return true;
}
