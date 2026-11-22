//! The engine this app starts, and nothing else.
//!
//! The engine (`api-server`, a PyInstaller --onefile build) used to be stopped
//! with `pkill -f api-server` (`taskkill /IM api-server.exe` on Windows). That
//! also killed every other process whose command line contains "api-server":
//! another installed copy of this app, or a different app that bundles an
//! engine of the same name (Lab Analysis Suite did, and lost its engine every
//! time this app quit). Now only the process this app spawned and the
//! processes below it are stopped, by pid.
//!
//! The engine also used to listen on the fixed port 8765 and the app trusted
//! whatever answered there. It now gets a free localhost port picked at launch
//! plus a per-launch token that it echoes in `/api/health`, so the app never
//! mistakes another program's server for its own.

use std::process::Command;

/// Kill the process `pid` and every process started below it.
///
/// `kill_root` stops the root itself (e.g. `CommandChild::kill`, which also
/// reaps it). A --onefile engine is a bootloader with the Python engine as
/// its child; SIGKILL on the bootloader cannot be forwarded, so the child
/// (and anything it started — R, Python scripts) must be stopped by pid.
pub fn kill_tree(pid: u32, kill_root: impl FnOnce()) {
    #[cfg(unix)]
    {
        // Collected first: once the root is gone its children are re-parented
        // and can no longer be found from its pid.
        let below = descendants_of(pid);
        kill_root();
        if !below.is_empty() {
            let mut cmd = Command::new("/bin/kill");
            cmd.arg("-9");
            for p in &below {
                cmd.arg(p.to_string());
            }
            let _ = cmd.output();
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // /T: this pid and the processes it started; nothing chosen by name.
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .output();
        kill_root();
    }
}

/// Every process below `pid` (children, grandchildren, ...).
#[cfg(unix)]
pub fn descendants_of(pid: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut queue = vec![pid];
    while let Some(p) = queue.pop() {
        let Ok(o) = Command::new("pgrep").arg("-P").arg(p.to_string()).output() else { continue };
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            if let Ok(c) = line.trim().parse::<u32>() {
                if c != pid && !out.contains(&c) {
                    out.push(c);
                    queue.push(c);
                }
            }
        }
        if out.len() > 4096 {
            break; // runaway guard
        }
    }
    out
}

/// A localhost port that is free right now (the OS picks it). The engine binds
/// it moments later; it is never a port another program is listening on.
pub fn free_local_port() -> std::io::Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(l.local_addr()?.port())
}

/// A per-launch token (32 hex chars). Not a secret: it only lets the app tell
/// its own engine from anything else that answers on the port.
pub fn new_launch_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut parts = [0u64; 2];
    for (i, part) in parts.iter_mut().enumerate() {
        // RandomState is seeded from the OS's randomness per process.
        let mut h = RandomState::new().build_hasher();
        h.write_u128(now);
        h.write_u32(std::process::id());
        h.write_usize(i);
        *part = h.finish();
    }
    format!("{:016x}{:016x}", parts[0], parts[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_port_is_bindable_and_not_8765() {
        let p = free_local_port().unwrap();
        assert_ne!(p, 0);
        // Ephemeral ports are far above 8765; the fixed port is never handed out.
        assert_ne!(p, 8765);
        let l = std::net::TcpListener::bind(("127.0.0.1", p)).expect("picked port is not free");
        drop(l);
    }

    #[test]
    fn launch_tokens_differ() {
        let a = new_launch_token();
        let b = new_launch_token();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}

#[cfg(all(test, unix))]
mod unix_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn alive(pid: u32) -> bool {
        // A zombie still answers kill -0; ask ps for its state instead.
        let o = Command::new("ps").args(["-o", "stat=", "-p", &pid.to_string()]).output().unwrap();
        let stat = String::from_utf8_lossy(&o.stdout).trim().to_string();
        !stat.is_empty() && !stat.starts_with('Z')
    }

    fn wait_until(mut f: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(5) {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    #[test]
    fn kills_the_tree_and_leaves_other_processes_alone() {
        // An unrelated process whose command line contains "api-server" (what
        // the old pkill pattern matched), standing in for another app's engine.
        let mut decoy = Command::new("sh").args(["-c", "sleep 30; : api-server", "api-server"]).spawn().unwrap();
        // The "engine": a shell (the bootloader) with a child that has a child
        // of its own (the Python engine and something it started).
        let mut engine = Command::new("sh")
            .args(["-c", "sh -c 'sleep 30 & wait' & sleep 30 & wait", "api-server"])
            .spawn()
            .unwrap();
        assert!(wait_until(|| descendants_of(engine.id()).len() >= 3), "tree did not start");
        let below = descendants_of(engine.id());
        let root = engine.id();

        kill_tree(root, || {
            let _ = engine.kill();
            let _ = engine.wait();
        });

        assert!(wait_until(|| !alive(root) && below.iter().all(|p| !alive(*p))), "tree survived: {below:?}");
        assert!(alive(decoy.id()), "the unrelated api-server process was killed");
        let _ = decoy.kill();
        let _ = decoy.wait();
    }

    /// The real engine: `MPF_ENGINE_BIN=/path/to/api-server-<triple> cargo
    /// test -- --ignored real_engine`. Starts it on a free port with a token,
    /// waits for /api/health to echo the token, then kill_tree must leave no
    /// process of it behind (a --onefile build is bootloader + Python child).
    #[test]
    #[ignore]
    fn real_engine_starts_on_free_port_and_stops_completely() {
        use std::io::{Read, Write};
        let bin = std::env::var("MPF_ENGINE_BIN").expect("set MPF_ENGINE_BIN");
        let port = free_local_port().unwrap();
        let token = new_launch_token();
        let mut engine = Command::new(&bin)
            .args(["--port", &port.to_string()])
            .env("MPF_LAUNCH_TOKEN", &token)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let health = || -> Option<String> {
            let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
            s.write_all(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").ok()?;
            let mut buf = String::new();
            s.read_to_string(&mut buf).ok()?;
            Some(buf)
        };
        let start = Instant::now();
        let mut reply = None;
        while start.elapsed() < Duration::from_secs(120) {
            if let Some(r) = health() {
                reply = Some(r);
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        // Stop it before any assertion, so a failing check can't orphan it.
        let root = engine.id();
        let below = descendants_of(root);
        eprintln!("engine pid {root}, below: {below:?}");
        kill_tree(root, || {
            let _ = engine.kill();
            let _ = engine.wait();
        });
        let gone = wait_until(|| !alive(root) && below.iter().all(|p| !alive(*p)));

        let reply = reply.expect("engine never answered");
        // Very old local builds have no /api/health (404); then readiness is all
        // that can be checked. Engines built before the token answer without one.
        if !reply.contains("Not Found") {
            assert!(reply.contains("\"status\":\"ok\""), "unexpected health reply: {reply}");
            if reply.contains("\"token\"") {
                assert!(reply.contains(&format!("\"token\":\"{token}\"")), "health did not echo token: {reply}");
            }
        }
        assert!(!below.is_empty(), "expected a --onefile bootloader with a child");
        assert!(gone, "engine left processes: {below:?}");
        assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok(), "port still held after stop");
    }

    #[test]
    fn killing_root_alone_would_orphan_the_child() {
        // Documents why kill_tree exists: SIGKILL on the root (what
        // CommandChild::kill does) leaves its child running.
        let mut engine = Command::new("sh").args(["-c", "sleep 30 & wait"]).spawn().unwrap();
        assert!(wait_until(|| !descendants_of(engine.id()).is_empty()), "child did not start");
        let below = descendants_of(engine.id());
        let _ = engine.kill();
        let _ = engine.wait();
        std::thread::sleep(Duration::from_millis(200));
        assert!(below.iter().all(|p| alive(*p)), "expected the orphan to survive a root-only kill");
        // Clean up by pid — only what this test started.
        let _ = Command::new("/bin/kill").arg("-9").args(below.iter().map(|p| p.to_string())).output();
    }
}
