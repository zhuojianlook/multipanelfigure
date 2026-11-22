/* utils/docIdentity: the last-seen table (one entry per path, newest first,
   capped, unreadable storage reads as empty), buildRef, each tab's on-disk
   uid with its change notifications, and isMissingFileError. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  DOC_REGISTRY_CAP, DOC_REGISTRY_KEY, buildRef, clearFileUid, cloudProviderOf, fileUidOf, forgetDocFile,
  isCloudDownloadError, isMissingFileError, loadFailureNotice,
  noteDocFile, onIdentityChange, registryPathsOf, registryUidAt, resetDocIdentityForTests, setDocRegistryStorage,
  setFileUid, type DocRegistryStorage,
} from "./docIdentity";

/** An in-memory storage (the table is injected, never the real localStorage). */
function memStorage(): DocRegistryStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (k) => data.get(k) ?? null,
    setItem: (k, v) => { data.set(k, String(v)); },
  };
}
const tableOf = (s: { data: Map<string, string> }) => JSON.parse(s.data.get(DOC_REGISTRY_KEY) || "null");

let mem: ReturnType<typeof memStorage>;
beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(1_000_000);
  resetDocIdentityForTests();
  mem = memStorage();
  setDocRegistryStorage(mem);
});
afterEach(() => {
  resetDocIdentityForTests();
  vi.useRealTimers();
});

describe("the last-seen table", () => {
  it("keeps one entry per path (the latest uid wins); a uid may have several paths, newest first", () => {
    noteDocFile("du_a", "/exp/a.mpf", "a");
    noteDocFile("du_b", "/exp/b.mpf");
    noteDocFile("du_a", "/exp/renamed.mpf");                // same document, seen at a new path
    expect(registryPathsOf("du_a")).toEqual(["/exp/renamed.mpf", "/exp/a.mpf"]);
    expect(registryUidAt("/exp/a.mpf")).toBe("du_a");
    noteDocFile("du_other", "/exp/a.mpf");                   // another document now lives at a.mpf
    expect(registryUidAt("/exp/a.mpf")).toBe("du_other");
    expect(registryPathsOf("du_a")).toEqual(["/exp/renamed.mpf"]);
    expect(tableOf(mem).entries.filter((e: { path: string }) => e.path === "/exp/a.mpf")).toHaveLength(1);
    // Seen again: it becomes the newest, even within the same millisecond.
    noteDocFile("du_a", "/exp/a2.mpf");
    noteDocFile("du_a", "/exp/renamed.mpf");
    expect(registryPathsOf("du_a")).toEqual(["/exp/renamed.mpf", "/exp/a2.mpf"]);
  });

  it("keeps the name it was seen under unless a new one is given; nothing for empty input", () => {
    noteDocFile("du_a", "/exp/a.mpf", "a");
    noteDocFile("du_a", "/exp/a.mpf");
    expect(tableOf(mem).entries[0]).toMatchObject({ uid: "du_a", path: "/exp/a.mpf", name: "a" });
    noteDocFile("", "/exp/x.mpf");
    noteDocFile("du_x", "");
    expect(tableOf(mem).entries).toHaveLength(1);
    expect(registryPathsOf(undefined)).toEqual([]);
    expect(registryUidAt(null)).toBeUndefined();
  });

  it("forgetDocFile drops one path", () => {
    noteDocFile("du_a", "/exp/a.mpf");
    noteDocFile("du_a", "/exp/a2.mpf");
    forgetDocFile("/exp/a.mpf");
    expect(registryPathsOf("du_a")).toEqual(["/exp/a2.mpf"]);
    forgetDocFile("/nowhere.mpf");                           // nothing to drop: no error
    expect(registryUidAt("/exp/a2.mpf")).toBe("du_a");
  });

  it("is capped: the oldest entries go first", () => {
    for (let i = 0; i < DOC_REGISTRY_CAP + 7; i++) noteDocFile(`du_${i}`, `/f/${i}.mpf`);
    const entries = tableOf(mem).entries as Array<{ uid: string }>;
    expect(entries).toHaveLength(DOC_REGISTRY_CAP);
    expect(registryUidAt("/f/0.mpf")).toBeUndefined();
    expect(registryUidAt("/f/6.mpf")).toBeUndefined();
    expect(registryUidAt("/f/7.mpf")).toBe("du_7");
    expect(entries[0].uid).toBe(`du_${DOC_REGISTRY_CAP + 6}`);
  });

  it("unreadable storage reads as empty, and a write that fails is dropped quietly", () => {
    mem.data.set(DOC_REGISTRY_KEY, "{not json");
    expect(registryPathsOf("du_a")).toEqual([]);
    mem.data.set(DOC_REGISTRY_KEY, JSON.stringify({ v: 1, entries: "nope" }));
    expect(registryUidAt("/exp/a.mpf")).toBeUndefined();
    // Entries of the wrong shape are skipped, the good ones kept.
    mem.data.set(DOC_REGISTRY_KEY, JSON.stringify({ v: 1, entries: [
      null, 3, { uid: "", path: "/x.mpf", t: 1 }, { uid: "du_a", path: 7, t: 1 }, { uid: "du_a", path: "/y.mpf", t: "x" },
      { uid: "du_a", path: "/exp/a.mpf", t: 5 },
    ] }));
    expect(registryPathsOf("du_a")).toEqual(["/exp/a.mpf"]);
    // setItem throws (storage full): no error, nothing changes.
    setDocRegistryStorage({ getItem: mem.getItem, setItem: () => { throw new Error("QuotaExceededError"); } });
    expect(() => noteDocFile("du_b", "/exp/b.mpf")).not.toThrow();
    expect(registryUidAt("/exp/b.mpf")).toBeUndefined();
    // No storage at all.
    setDocRegistryStorage(null);
    expect(() => noteDocFile("du_b", "/exp/b.mpf")).not.toThrow();
    expect(registryPathsOf("du_a")).toEqual([]);
  });

  it("lives in localStorage by default", () => {
    setDocRegistryStorage(undefined);
    localStorage.removeItem(DOC_REGISTRY_KEY);
    noteDocFile("du_a", "/exp/a.mpf");
    expect(JSON.parse(localStorage.getItem(DOC_REGISTRY_KEY)!).entries[0]).toMatchObject({ uid: "du_a", path: "/exp/a.mpf" });
    localStorage.removeItem(DOC_REGISTRY_KEY);
  });
});

describe("buildRef", () => {
  it("a ref without a uid checks its path only", () => {
    noteDocFile("du_a", "/elsewhere/a.mpf");
    expect(buildRef("k", "/exp/a.mpf", null, ["/other"])).toEqual({ key: "k", path: "/exp/a.mpf" });
  });

  it("the other paths the uid was seen at are candidates; extra folders not already implied are dirs", () => {
    noteDocFile("du_a", "/exp/a.mpf");
    noteDocFile("du_a", "/old/place/a.mpf");
    noteDocFile("du_b", "/b/b.mpf");
    const ref = buildRef("0", "/exp/a.mpf", "du_a", ["/exp", "/old/place", "/open/doc", "/open/doc", ""]);
    expect(ref).toEqual({
      key: "0", path: "/exp/a.mpf", uid: "du_a",
      candidates: ["/old/place/a.mpf"],
      dirs: ["/open/doc"],
    });
    expect(buildRef("1", "/exp/a.mpf", "du_unknown")).toEqual({ key: "1", path: "/exp/a.mpf", uid: "du_unknown" });
  });
});

describe("each tab's on-disk uid", () => {
  it("undefined until known, null when the file names none; cleared per tab and by the test reset", () => {
    expect(fileUidOf("doc_a")).toBeUndefined();
    setFileUid("doc_a", null, "/exp/a.mpf");
    expect(fileUidOf("doc_a")).toBeNull();
    setFileUid("doc_a", "du_a", "/exp/a.mpf");
    setFileUid("doc_b", "du_b", "/exp/b.mpf");
    expect(fileUidOf("doc_a")).toBe("du_a");
    clearFileUid("doc_a");
    expect(fileUidOf("doc_a")).toBeUndefined();
    expect(fileUidOf("doc_b")).toBe("du_b");
    resetDocIdentityForTests();
    expect(fileUidOf("doc_b")).toBeUndefined();
  });

  it("listeners hear each new identification, not a repeat or a 'none'; they can unsubscribe", () => {
    const heard = vi.fn();
    const stop = onIdentityChange(heard);
    setFileUid("doc_a", null, "/exp/a.mpf");
    expect(heard).not.toHaveBeenCalled();
    setFileUid("doc_a", "du_a", "/exp/a.mpf");
    setFileUid("doc_a", "du_a", "/exp/a.mpf");               // nothing new
    expect(heard).toHaveBeenCalledTimes(1);
    expect(heard).toHaveBeenLastCalledWith({ docId: "doc_a", uid: "du_a", path: "/exp/a.mpf" });
    setFileUid("doc_a", "du_a", "/exp/renamed.mpf");         // same file, found at a new path
    setFileUid("doc_a", "du_a2", "/exp/renamed.mpf");        // re-identified by a save
    expect(heard).toHaveBeenCalledTimes(3);
    // A listener that throws doesn't stop the others.
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const stopBad = onIdentityChange(() => { throw new Error("boom"); });
    setFileUid("doc_b", "du_b", "/exp/b.mpf");
    expect(heard).toHaveBeenCalledTimes(4);
    stopBad();
    err.mockRestore();
    stop();
    setFileUid("doc_c", "du_c", "/exp/c.mpf");
    expect(heard).toHaveBeenCalledTimes(4);
  });
});

describe("isMissingFileError", () => {
  it("recognises the sidecar's 404 for a path with no file", () => {
    expect(isMissingFileError(new Error('API error: "Project file not found: /exp/a.mpf"'))).toBe(true);
    expect(isMissingFileError("API error: \"Project file not found: x\"")).toBe(true);
    expect(isMissingFileError(new Error("disk full"))).toBe(false);
    expect(isMissingFileError(new Error('API error: "Image \'x\' not found"'))).toBe(false);
    expect(isMissingFileError(undefined)).toBe(false);
  });
});

describe("files stored online only", () => {
  const GD = "/Users/u/Library/CloudStorage/GoogleDrive-u@x.com/My Drive/Lab Data/DeepRed Staining.mpf";
  const cloudErr = new Error(`API error: "Cloud file could not be downloaded: ${GD} (Resource deadlock avoided)"`);

  it("recognises the sidecar's cloud download failure, and never as a missing file", () => {
    expect(isCloudDownloadError(cloudErr)).toBe(true);
    expect(isMissingFileError(cloudErr)).toBe(false);
    expect(isCloudDownloadError(new Error('API error: "Project file not found: /a.mpf"'))).toBe(false);
    expect(isCloudDownloadError(undefined)).toBe(false);
  });

  it("names the cloud service from the synced folder's path", () => {
    expect(cloudProviderOf(GD)).toBe("Google Drive");
    expect(cloudProviderOf("/Users/u/Library/CloudStorage/Dropbox/GitHub/x.mpf")).toBe("Dropbox");
    expect(cloudProviderOf("/Users/u/Dropbox (Lab)/x.mpf")).toBe("Dropbox");
    expect(cloudProviderOf("/Users/u/Library/CloudStorage/OneDrive-Uni/x.mpf")).toBe("OneDrive");
    expect(cloudProviderOf("C:\\Users\\u\\OneDrive - Uni\\x.mpf")).toBe("OneDrive");
    expect(cloudProviderOf("/Users/u/Library/Mobile Documents/com~apple~CloudDocs/x.mpf")).toBe("iCloud Drive");
    expect(cloudProviderOf("G:\\My Drive\\x.mpf")).toBe("Google Drive");
    expect(cloudProviderOf("/Users/u/Documents/x.mpf")).toBeNull();
    expect(cloudProviderOf(null)).toBeNull();
  });

  it("the notice says the file is online only and what to check — not that it was moved or deleted", () => {
    const n = loadFailureNotice("DeepRed Staining", GD, cloudErr);
    expect(n.title).toBe("Could not download the file");
    expect(n.body).toContain('"DeepRed Staining" is stored online only in Google Drive');
    expect(n.body).toContain("Check that Google Drive is running and that you are online");
    expect(n.body).not.toMatch(/moved or deleted/);
    const anywhere = loadFailureNotice("x", "/Volumes/NAS/x.mpf", cloudErr);
    expect(anywhere.body).toContain("Check that your cloud storage app is running");
  });

  it("any other failure says what the sidecar reported", () => {
    const n = loadFailureNotice("x", "/a/x.mpf", new Error('API error: "Could not read /a/x.mpf: Permission denied"'));
    expect(n).toEqual({ title: "Load failed", body: 'Could not load "x".\n\nCould not read /a/x.mpf: Permission denied' });
    // Not the sidecar's JSON (a plain 500, a dropped connection): the old hint.
    expect(loadFailureNotice("x", "/a/x.mpf", new SyntaxError("Unexpected token")).body)
      .toMatch(/could not be read\. It may be damaged, or it may have been moved or deleted/);
  });
});
