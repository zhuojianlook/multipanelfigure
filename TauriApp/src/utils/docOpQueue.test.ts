import { describe, it, expect } from "vitest";
import { serializeDocOp, whenDocOpsIdle, isStateCacheMiss } from "./docOpQueue";

function deferred<T = void>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("doc-op queue", () => {
  it("runs ops strictly one at a time, in order", async () => {
    const log: string[] = [];
    const d = deferred();
    const a = serializeDocOp(async () => { log.push("a:start"); await d.promise; log.push("a:end"); });
    const b = serializeDocOp(async () => { log.push("b"); });
    await Promise.resolve();
    expect(log).toEqual(["a:start"]);
    d.resolve();
    await Promise.all([a, b]);
    expect(log).toEqual(["a:start", "a:end", "b"]);
  });

  it("a failed op does not break the chain", async () => {
    const failed = serializeDocOp(async () => { throw new Error("x"); });
    const next = serializeDocOp(async () => "ok");
    await expect(failed).rejects.toThrow("x");
    await expect(next).resolves.toBe("ok");
  });

  it("whenDocOpsIdle waits for ops queued while it waits", async () => {
    const d1 = deferred();
    const d2 = deferred();
    void serializeDocOp(() => d1.promise);
    let idle = false;
    const w = whenDocOpsIdle().then(() => { idle = true; });
    void serializeDocOp(() => d2.promise);   // queued after the wait began
    d1.resolve();
    await new Promise((r) => setTimeout(r, 0));
    expect(idle).toBe(false);
    d2.resolve();
    await w;
    expect(idle).toBe(true);
  });

  it("recognises only a genuine activate-state cache miss", () => {
    expect(isStateCacheMiss(new Error('API error: "No cached state for that document."'))).toBe(true);
    expect(isStateCacheMiss(new Error('API error: {"code":"no_cached_state"}'))).toBe(true);
    expect(isStateCacheMiss(new Error('API error: "Document state is temporarily swapped; retry."'))).toBe(false);
    expect(isStateCacheMiss(new Error("Request failed: connection refused"))).toBe(false);
  });
});
