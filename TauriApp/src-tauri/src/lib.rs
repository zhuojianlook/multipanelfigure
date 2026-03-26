use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::process::CommandChild;
use std::sync::{Arc, Mutex};

#[tauri::command]
fn get_sidecar_port(state: tauri::State<'_, SidecarPort>) -> u16 {
    state.0
}

#[tauri::command]
fn get_sidecar_error(state: tauri::State<'_, SidecarError>) -> Option<String> {
    state.0.lock().unwrap().clone()
}

struct SidecarPort(u16);
struct SidecarError(Arc<Mutex<Option<String>>>);
struct SidecarChild(Arc<Mutex<Option<CommandChild>>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_shell::init())
    .plugin(tauri_plugin_updater::Builder::new().build())
    .plugin(tauri_plugin_process::init())
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      let sidecar_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
      let sidecar_child: Arc<Mutex<Option<CommandChild>>> = Arc::new(Mutex::new(None));

      // Launch sidecar in production; in dev, assume it's already running
      let port: u16;
      if cfg!(debug_assertions) {
        port = 8765;
      } else {
        // On macOS, remove quarantine attribute from sidecar before spawning.
        // This prevents Gatekeeper from blocking the sidecar on fresh installs
        // when the user has already approved the main app via right-click → Open.
        #[cfg(target_os = "macos")]
        {
          if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
              let sidecar_name = format!("api-server-{}-apple-darwin", std::env::consts::ARCH);
              let sidecar_path = exe_dir.join(&sidecar_name);
              eprintln!("[setup] Removing quarantine from sidecar: {:?}", sidecar_path);
              let _ = std::process::Command::new("xattr")
                .args(["-dr", "com.apple.quarantine", &sidecar_path.to_string_lossy().to_string()])
                .output();
              let _ = std::process::Command::new("chmod")
                .args(["+x", &sidecar_path.to_string_lossy().to_string()])
                .output();
            }
          }
        }

        let sidecar_result = app.shell().sidecar("api-server");
        match sidecar_result {
          Ok(cmd) => {
            let cmd = cmd.args(["--port", "8765"]);
            match cmd.spawn() {
              Ok((mut rx, child)) => {
                // Store child handle so we can kill it on app exit
                *sidecar_child.lock().unwrap() = Some(child);
                // Capture sidecar stdout/stderr in background
                let err_clone = sidecar_error.clone();
                tauri::async_runtime::spawn(async move {
                  while let Some(event) = rx.recv().await {
                    match event {
                      CommandEvent::Stderr(line) => {
                        let msg = String::from_utf8_lossy(&line).to_string();
                        eprintln!("[sidecar stderr] {}", msg);
                        let mut e = err_clone.lock().unwrap();
                        let current = e.get_or_insert_with(String::new);
                        if current.len() < 2000 {
                          current.push_str(&msg);
                          current.push('\n');
                        }
                      }
                      CommandEvent::Stdout(line) => {
                        eprintln!("[sidecar stdout] {}", String::from_utf8_lossy(&line));
                      }
                      CommandEvent::Terminated(payload) => {
                        let msg = format!("Sidecar exited with code: {:?}, signal: {:?}", payload.code, payload.signal);
                        eprintln!("{}", msg);
                        let mut e = err_clone.lock().unwrap();
                        let current = e.get_or_insert_with(String::new);
                        current.push_str(&msg);
                      }
                      _ => {}
                    }
                  }
                });
              }
              Err(e) => {
                let msg = format!("Failed to spawn sidecar: {}", e);
                eprintln!("{}", msg);
                *sidecar_error.lock().unwrap() = Some(msg);
              }
            }
          }
          Err(e) => {
            let msg = format!("Failed to find sidecar binary: {}", e);
            eprintln!("{}", msg);
            *sidecar_error.lock().unwrap() = Some(msg);
          }
        }
        port = 8765;
      }

      app.manage(SidecarPort(port));
      app.manage(SidecarError(sidecar_error));
      app.manage(SidecarChild(sidecar_child));
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![get_sidecar_port, get_sidecar_error])
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      // Kill sidecar when app exits
      if let tauri::RunEvent::Exit = event {
        if let Some(state) = app_handle.try_state::<SidecarChild>() {
          if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.take() {
              eprintln!("[cleanup] Killing sidecar process");
              let _ = child.kill();
            }
          }
        }
      }
    });
}
