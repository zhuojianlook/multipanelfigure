/* ──────────────────────────────────────────────────────────
   Visual audit — captures screenshots of every UI section
   for manual review. Not assertion-based; just captures.
   ────────────────────────────────────────────────────────── */

import { test } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOTS = path.join(__dirname, "screenshots");

test.describe("Visual Audit — Full App", () => {
  test("main app at 1280x800", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(SHOTS, "01-main-1280x800.png"), fullPage: false });
  });

  test("main app at 900x700", async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 700 });
    await page.goto("/");
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(SHOTS, "02-main-900x700.png"), fullPage: false });
  });

  test("main app full page scroll", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(SHOTS, "03-main-fullpage.png"), fullPage: true });
  });
});

test.describe("Visual Audit — EditPanelDialog Tabs", () => {
  async function openDialog(page: import("@playwright/test").Page) {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    const editBtn = page.locator('button[aria-label="Edit panel"]').first();
    await editBtn.click({ force: true });
    await page.waitForTimeout(600);
  }

  test("tab 0 — Crop/Resize", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Crop/Resize" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "10-dialog-crop-resize.png") });
  });

  test("tab 1 — Adjustments", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Adjustments" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "11-dialog-image-adjustments.png") });
  });

  test("tab 2 — Labels (empty)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Labels" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "12-dialog-labels-empty.png") });
  });

  test("tab 2 — Labels (with label added)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Labels" }).click();
    await page.waitForTimeout(300);
    await page.getByRole("button", { name: "Add Label" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "13-dialog-labels-with-entry.png") });
  });

  test("tab 3 — Scale Bar (disabled)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Scale Bar" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "14-dialog-scalebar-disabled.png") });
  });

  test("tab 3 — Scale Bar (enabled)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Scale Bar" }).click();
    await page.waitForTimeout(300);
    await page.locator(".MuiSwitch-root").first().click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "15-dialog-scalebar-enabled.png") });
  });

  test("tab 4 — Symbols (empty)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Symbols" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "16-dialog-symbols-empty.png") });
  });

  test("tab 4 — Symbols (with symbol added)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Symbols" }).click();
    await page.waitForTimeout(300);
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "17-dialog-symbols-with-entry.png") });
  });

  test("tab 5 — Zoom Inset (disabled)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Zoom Inset" }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "18-dialog-zoom-disabled.png") });
  });

  test("tab 5 — Zoom Inset (enabled)", async ({ page }) => {
    await openDialog(page);
    await page.getByRole("tab", { name: "Zoom Inset" }).click();
    await page.waitForTimeout(300);
    await page.locator(".MuiSwitch-root").first().click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(SHOTS, "19-dialog-zoom-enabled.png") });
  });
});

test.describe("Visual Audit — Dialog at narrow viewport", () => {
  test("dialog at 800x600", async ({ page }) => {
    await page.setViewportSize({ width: 800, height: 600 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    const editBtn = page.locator('button[aria-label="Edit panel"]').first();
    await editBtn.click({ force: true });
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(SHOTS, "20-dialog-narrow-800x600.png") });
  });
});
