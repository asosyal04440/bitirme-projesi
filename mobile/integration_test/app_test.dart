import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:sinav_okuyucu/main.dart';
import 'package:sinav_okuyucu/src/rust/frb_generated.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => RustLib.init());

  testWidgets('the app starts on the course list', (tester) async {
    await tester.pumpWidget(const SinavOkuyucuApp());
    await tester.pumpAndSettle();
    expect(find.text('Dersler'), findsOneWidget);
    expect(find.text('Ders ekle'), findsOneWidget);
  });
}
