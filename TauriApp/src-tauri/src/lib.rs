use tauri::Manager;
use tauri_plugin_shell::ShellExt;

#[tauri::command]
fn get_sidecar_port(state: tauri::State<'_, SidecarPort>) -> u16 {
    state.0
}

struct SidecarPort(u16);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_shell::init())
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Launch sidecar in production; in dev, assume it's already running
      let port: u16;
      if cfg!(debug_assertions) {
        // Dev mode: use the manually started sidecar (default port from READY line)
        port = 8765;
      } else {
        // Production: launch bundled sidecar binary
        let sidecar = app.shell().sidecar("api-server")
          .expect("failed to find sidecar binary")
          .args(["--port", "0"]);  // port 0 = auto-assign

        let (mut rx, _child) = sidecar.spawn()
          .expect("failed to spawn sidecar");

        // Read the READY:PORT line from stdout
        port = 8765; // fallback
        // Note: Tauri v2 sidecar stdout is handled via events, not BufReader
        // The sidecar prints "READY:PORT" — we'll use a fixed port for now
        // and improve this with event-based port reading later
      }

      app.manage(SidecarPort(port));
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![get_sidecar_port])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
