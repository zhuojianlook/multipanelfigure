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

/// Proxy HTTP requests to the sidecar through Rust, bypassing WebView restrictions
#[tauri::command]
async fn proxy_request(
    method: String,
    path: String,
    body: Option<String>,
    state: tauri::State<'_, SidecarPort>,
) -> Result<String, String> {
    let port = state.0;
    let url = format!("http://127.0.0.1:{}{}", port, path);
    let client = reqwest::Client::new();

    let req = match method.to_uppercase().as_str() {
        "GET" => client.get(&url),
        "POST" => {
            let mut r = client.post(&url);
            if let Some(b) = body {
                r = r.header("Content-Type", "application/json").body(b);
            }
            r
        }
        "PUT" => {
            let mut r = client.put(&url);
            if let Some(b) = body {
                r = r.header("Content-Type", "application/json").body(b);
            }
            r
        }
        "PATCH" => {
            let mut r = client.patch(&url);
            if let Some(b) = body {
                r = r.header("Content-Type", "application/json").body(b);
            }
            r
        }
        "DELETE" => client.delete(&url),
        _ => return Err(format!("Unsupported method: {}", method)),
    };

    let resp = req.send().await.map_err(|e| format!("Request failed: {}", e))?;
    let text = resp.text().await.map_err(|e| format!("Failed to read response: {}", e))?;
    Ok(text)
}

/// Kill the sidecar process — called before update restart
#[tauri::command]
async fn kill_sidecar(state: tauri::State<'_, SidecarChild>) -> Result<(), String> {
    kill_sidecar_process(&state.0);
    Ok(())
}

/// Helper to kill sidecar and any child processes
fn kill_sidecar_process(child_mutex: &Arc<Mutex<Option<CommandChild>>>) {
    if let Ok(mut guard) = child_mutex.lock() {
        if let Some(child) = guard.take() {
            eprintln!("[cleanup] Killing sidecar process");
            let _ = child.kill();
        }
    }
    // On Windows, PyInstaller --onefile spawns a child process that may survive.
    // Kill any remaining api-server processes.
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "api-server.exe", "/T"])
            .output();
    }
    // On macOS/Linux, kill any remaining api-server processes
    #[cfg(not(target_os = "windows"))]
    {
        let _ = std::process::Command::new("pkill")
            .args(["-f", "api-server"])
            .output();
    }
}

/// Proxy file uploads to the sidecar through Rust
#[tauri::command]
async fn proxy_upload(
    path: String,
    files: Vec<FileData>,
    field_name: String,
    state: tauri::State<'_, SidecarPort>,
) -> Result<String, String> {
    let port = state.0;
    let url = format!("http://127.0.0.1:{}{}", port, path);
    let client = reqwest::Client::new();

    let mut form = reqwest::multipart::Form::new();
    for file in files {
        let decoded = base64_decode(&file.data).map_err(|e| format!("Base64 decode error: {}", e))?;
        let part = reqwest::multipart::Part::bytes(decoded)
            .file_name(file.name)
            .mime_str("application/octet-stream")
            .map_err(|e| format!("MIME error: {}", e))?;
        form = form.part(field_name.clone(), part);
    }

    let resp = client.post(&url).multipart(form).send().await
        .map_err(|e| format!("Upload request failed: {}", e))?;
    let text = resp.text().await
        .map_err(|e| format!("Failed to read upload response: {}", e))?;
    Ok(text)
}

#[derive(serde::Deserialize)]
struct FileData {
    name: String,
    data: String,  // base64 encoded
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    // Simple base64 decoder
    let lookup = |c: u8| -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(format!("Invalid base64 char: {}", c as char)),
        }
    };
    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        if chunk.len() < 4 { break; }
        let a = lookup(chunk[0])?;
        let b = lookup(chunk[1])?;
        let c = lookup(chunk[2])?;
        let d = lookup(chunk[3])?;
        result.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' { result.push((b << 4) | (c >> 2)); }
        if chunk[3] != b'=' { result.push((c << 6) | d); }
    }
    Ok(result)
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
        // On macOS, remove quarantine attribute from the entire .app bundle.
        // PyInstaller --onefile binaries extract to /tmp at runtime, and macOS
        // propagates the quarantine flag to extracted files. Removing it from
        // the whole bundle prevents Gatekeeper from blocking the sidecar.
        #[cfg(target_os = "macos")]
        {
          if let Ok(exe_path) = std::env::current_exe() {
            // exe is at .app/Contents/MacOS/app — walk up to .app
            if let Some(macos_dir) = exe_path.parent() {
              // Remove quarantine from sidecar binary
              let sidecar_name = format!("api-server-{}-apple-darwin", std::env::consts::ARCH);
              let sidecar_path = macos_dir.join(&sidecar_name);
              eprintln!("[setup] Removing quarantine from sidecar: {:?}", sidecar_path);
              let _ = std::process::Command::new("xattr")
                .args(["-cr", &sidecar_path.to_string_lossy().to_string()])
                .output();
              let _ = std::process::Command::new("chmod")
                .args(["+x", &sidecar_path.to_string_lossy().to_string()])
                .output();

              // Also remove quarantine from the entire .app bundle
              if let Some(contents_dir) = macos_dir.parent() {
                if let Some(app_dir) = contents_dir.parent() {
                  eprintln!("[setup] Removing quarantine from app bundle: {:?}", app_dir);
                  let _ = std::process::Command::new("xattr")
                    .args(["-cr", &app_dir.to_string_lossy().to_string()])
                    .output();
                }
              }
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
    .invoke_handler(tauri::generate_handler![get_sidecar_port, get_sidecar_error, proxy_request, proxy_upload, kill_sidecar])
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      match event {
        tauri::RunEvent::Exit => {
          // Kill sidecar when app exits
          if let Some(state) = app_handle.try_state::<SidecarChild>() {
            kill_sidecar_process(&state.0);
          }
        }
        tauri::RunEvent::ExitRequested { .. } => {
          // Also kill sidecar when exit is requested (window close)
          if let Some(state) = app_handle.try_state::<SidecarChild>() {
            kill_sidecar_process(&state.0);
          }
        }
        _ => {}
      }
    });
}
