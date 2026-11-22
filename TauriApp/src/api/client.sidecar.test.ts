/* The app talks only to the engine IT started: the port comes from Tauri
   (a free port picked at launch, not a fixed 8765) and /api/health must echo
   this launch's token, so another program answering there is never ours. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));

type ClientModule = typeof import("./client");

async function freshClient(): Promise<ClientModule> {
  vi.resetModules();
  return import("./client");
}

/** Tauri reports `info`; proxied /api/health answers `health`. */
function tauri(info: unknown, health: unknown) {
  invokeMock.mockImplementation(async (cmd: string, args: { path?: string }) => {
    if (cmd === "get_sidecar_info") {
      if (info instanceof Error) throw info;
      return info;
    }
    if (cmd === "proxy_request" && args.path === "/api/health") return JSON.stringify(health);
    throw new Error(`unexpected ${cmd}`);
  });
}

beforeEach(() => {
  invokeMock.mockReset();
  (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
});

afterEach(() => {
  delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
});

describe("sidecar identity", () => {
  it("adopts the port Tauri started the engine on and accepts its own token", async () => {
    tauri({ port: 54321, token: "abc" }, { status: "ok", token: "abc" });
    const c = await freshClient();
    expect(await c.checkHealth()).toBe(true);
    expect(c.getApiBase()).toBe("http://127.0.0.1:54321");
  });

  it("rejects a server that answers on the port with another token", async () => {
    tauri({ port: 54321, token: "abc" }, { status: "ok", token: "someone-else" });
    const c = await freshClient();
    expect(await c.checkHealth()).toBe(false);
    expect(c.lastHealthError).toMatch(/not this app's engine/);
  });

  it("rejects a server with no token at all (e.g. another app's engine)", async () => {
    tauri({ port: 54321, token: "abc" }, { status: "ok" });
    const c = await freshClient();
    expect(await c.checkHealth()).toBe(false);
  });

  it("dev build (no token) accepts any healthy engine on the default port", async () => {
    tauri({ port: 8765, token: null }, { status: "ok" });
    const c = await freshClient();
    expect(await c.checkHealth()).toBe(true);
    expect(c.getApiBase()).toBe("http://127.0.0.1:8765");
  });

  it("keeps defaults when the backend has no get_sidecar_info command", async () => {
    tauri(new Error("unknown command"), { status: "ok" });
    const c = await freshClient();
    expect(await c.checkHealth()).toBe(true);
    expect(c.getApiBase()).toBe("http://127.0.0.1:8765");
  });
});

describe("healthIsOurs", () => {
  it("needs status ok and, when expected, the matching token", async () => {
    const { healthIsOurs } = await freshClient();
    expect(healthIsOurs({ status: "ok", token: "t" }, "t")).toBe(true);
    expect(healthIsOurs({ status: "ok", token: "x" }, "t")).toBe(false);
    expect(healthIsOurs({ status: "ok" }, null)).toBe(true);
    expect(healthIsOurs({ status: "down", token: "t" }, "t")).toBe(false);
  });
});
