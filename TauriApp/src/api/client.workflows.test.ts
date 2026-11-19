/* Wire contract of the saved-Analysis endpoints (save carry, workflows
   write/read) — what the sidecar receives and how its answers map. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./client";

let fetchMock: ReturnType<typeof vi.fn>;

function respond(body: unknown) {
  fetchMock.mockResolvedValueOnce({ text: async () => JSON.stringify(body) });
}

function sent(i = 0): { url: string; body: Record<string, unknown> } {
  const [url, init] = fetchMock.mock.calls[i] as [string, { body: string }];
  return { url, body: JSON.parse(init.body) };
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api.saveProject carry", () => {
  it("omits carry_from for a plain re-save", async () => {
    respond({ ok: true, path: "/a.mpf" });
    await api.saveProject("/a.mpf");
    expect(sent().url).toMatch(/\/api\/project\/save$/);
    expect(sent().body).toEqual({ path: "/a.mpf", analysis: null });
  });

  it("sends the previous file, or \"\" (carry nothing) for a document that never had one", async () => {
    respond({ ok: true });
    respond({ ok: true });
    await api.saveProject("/b.mpf", { carryFrom: "/a.mpf" });
    await api.saveProject("/b.mpf", { carryFrom: null });
    expect(sent(0).body).toEqual({ path: "/b.mpf", analysis: null, carry_from: "/a.mpf" });
    expect(sent(1).body).toEqual({ path: "/b.mpf", analysis: null, carry_from: "" });
  });

  it("passes a legacy analysis payload through", async () => {
    respond({ ok: true });
    const analysis = { manifest: { tabs: [] }, plots: {}, tables: {} };
    await api.saveProject("/a.mpf", { analysis });
    expect(sent().body).toEqual({ path: "/a.mpf", analysis });
  });
});

describe("api.writeWorkflows", () => {
  const payload = {
    manifest: { format: "mpf-workflows" as const, version: 1, workflows: [] },
    blobs: { b1: { kind: "png" as const, data: "AAAA" }, b2: { kind: "csv" as const } },
    uploads: ["stack.tif"],
  };

  it("posts path + payload and returns the result", async () => {
    respond({ ok: true, size_bytes: 10, workflows_bytes: 4, missing_uploads: [], fast_path: true });
    const r = await api.writeWorkflows("/a.mpf", payload);
    expect(sent().url).toMatch(/\/api\/project\/workflows\/write$/);
    expect(sent().body).toEqual({ path: "/a.mpf", ...payload });
    expect(r).toEqual({ ok: true, size_bytes: 10, workflows_bytes: 4, missing_uploads: [], fast_path: true });
  });

  it("maps a 409 for kept blobs the file lacks to {ok: false, missing}", async () => {
    respond({ detail: { missing: ["b2"] } });
    expect(await api.writeWorkflows("/a.mpf", payload)).toEqual({ ok: false, missing: ["b2"] });
  });

  it("maps a newer-format file to {ok: false, unsupported}", async () => {
    respond({ detail: { unsupported: true, version: 2 } });
    expect(await api.writeWorkflows("/a.mpf", payload)).toEqual({ ok: false, unsupported: true, version: 2 });
  });

  it("rejects on any other error", async () => {
    respond({ detail: "Project file not found: /a.mpf" });
    await expect(api.writeWorkflows("/a.mpf", payload)).rejects.toThrow(/not found/);
  });
});

describe("api.readWorkflows", () => {
  it("posts the path and returns the section", async () => {
    const section = {
      manifest: null, blobs: {}, unsupported: false, version: null,
      uploads: [], renamed: {}, missing_uploads: [], failed_uploads: [],
    };
    respond(section);
    expect(await api.readWorkflows("/a.mpf")).toEqual(section);
    expect(sent().url).toMatch(/\/api\/project\/workflows\/read$/);
    expect(sent().body).toEqual({ path: "/a.mpf" });
  });

  it("rejects when the sidecar reports an error", async () => {
    respond({ detail: "Project file not found: /gone.mpf" });
    await expect(api.readWorkflows("/gone.mpf")).rejects.toThrow(/not found/);
  });
});
