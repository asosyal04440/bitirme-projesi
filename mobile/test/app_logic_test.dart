import 'package:flutter_test/flutter_test.dart';
import 'package:sinav_okuyucu/app_state.dart';
import 'package:sinav_okuyucu/review_labels.dart';
import 'package:sinav_okuyucu/src/rust/api/reader.dart';

void main() {
  test('roster accepts semicolon, comma and tab separated lines', () {
    final roster = parseRoster(
      'Numara;Ad Soyad\n10000000000;Ayşe Yılmaz\n10000000001,Mehmet Kaya\n10000000002\tCan Demir\n\nbozuk satır\n',
    );
    expect(roster.map((s) => s.number), [
      '10000000000',
      '10000000001',
      '10000000002',
    ]);
    expect(roster.first.name, 'Ayşe Yılmaz');
  });

  test('review issues are explained in Turkish', () {
    expect(
      reviewText(
        const IssueResult(reason: 'low_confidence', field: 'question 2 score'),
      ),
      'S2 puanı emin olunamadan okundu.',
    );
    expect(
      reviewText(
        const IssueResult(reason: 'total_mismatch', field: 'declared total'),
      ),
      contains('toplam'),
    );
    expect(
      reviewText(
        const IssueResult(reason: 'roster_near_match', field: 'student number'),
      ),
      contains('tek hane'),
    );
  });
}
