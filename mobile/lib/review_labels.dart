import 'src/rust/api/reader.dart';

/// Turkish explanation of a review issue reported by the reader.
String reviewText(IssueResult issue) {
  final field = fieldText(issue.field);
  switch (issue.reason) {
    case 'low_confidence':
      return '$field emin olunamadan okundu.';
    case 'blank_field':
      return '$field boş görünüyor.';
    case 'ambiguous_corrected_score':
      return '$field üzerinde düzeltme var; son yazılan değer kontrol edilmeli.';
    case 'total_mismatch':
      return 'Kâğıttaki toplam not, soru puanlarının toplamıyla tutmuyor.';
    case 'score_exceeds_maximum':
      return '$field tam puandan büyük okundu.';
    case 'invalid_candidate':
      return '$field geçerli bir sayı olarak okunamadı.';
    case 'roster_near_match':
      return 'Öğrenci numarası tek hane farkla listedeki bir öğrenciye uyuyor; doğrulayın.';
    case 'unresolved_student_identity':
      return 'Öğrenci belirlenemedi; numarayı kontrol edin.';
    case 'conflicting_question_score':
      return '$field için çelişen değerler var.';
    case 'page_failed':
      return 'Kâğıt işlenemedi. Fotoğrafı yeniden çekin.';
    default:
      return '$field kontrol edilmeli.';
  }
}

String fieldText(String field) {
  final question = RegExp(r'question (\w+) score').firstMatch(field);
  if (question != null) {
    // The domain normalises `S3` to `3`; show it as printed on the sheet.
    final id = question.group(1)!;
    return '${RegExp(r'^\d+$').hasMatch(id) ? 'S$id' : id} puanı';
  }
  if (field.contains('declared total')) return 'Toplam not';
  if (field.contains('student number')) return 'Öğrenci numarası';
  if (field.contains('student')) return 'Öğrenci bilgisi';
  if (field.contains('page outline')) return 'Kâğıdın konumu';
  if (field.contains('score')) return 'Soru puanı';
  return 'Bu alan';
}
