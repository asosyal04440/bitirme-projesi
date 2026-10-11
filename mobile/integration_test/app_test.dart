import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:sinav_okuyucu/main.dart';
import 'package:sinav_okuyucu/src/rust/frb_generated.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => RustLib.init());

  testWidgets('the app starts on the exam list', (tester) async {
    await tester.pumpWidget(const SinavOkuyucuApp());
    await tester.pumpAndSettle();
    expect(find.text('Sınavlar'), findsOneWidget);
    expect(find.text('Word dosyasından sınav ekle'), findsOneWidget);
  });
}
