pub mod camera;
pub mod definition;
pub mod exam;
pub mod reader;
pub mod scanner;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
