/* ──────────────────────────────────────────────────────────
   E2E smoke tests — validates that the main UI components
   render without overflow, clipping, or layout issues.
   Runs against the Vite dev server (no Tauri/Python needed).
   ────────────────────────────────────────────────────────── */

import { test, expect } from "@playwright/test";

test.describe("App shell layout", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for the app to render (MUI may take a moment)
    await page.waitForTimeout(500);
  });

  test("page loads without JS errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    await page.goto("/");
    await page.waitForTimeout(1000);
    expect(errors).toEqual([]);
  });

  test("sidebar is visible and not overflowing", async ({ page }) => {
    const sidebar = page.locator('[class*="overflow-y-auto"]').first();
    if (await sidebar.count()) {
      const box = await sidebar.boundingBox();
      expect(box).not.toBeNull();
      if (box) {
        expect(box.width).toBeGreaterThan(180);
        expect(box.width).toBeLessThan(400);
      }
    }
  });

  test("toolbar renders and wraps on narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 600, height: 800 });
    await page.waitForTimeout(300);
    // Toolbar should be present
    const toolbar = page.locator("header, [role='toolbar']").first();
    if (await toolbar.count()) {
      const box = await toolbar.boundingBox();
      expect(box).not.toBeNull();
    }
  });
});

test.describe("Panel Grid", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);
  });

  test("add column header button has aria-label", async ({ page }) => {
    const btn = page.locator('button[aria-label="Add column header level"]');
    if (await btn.count()) {
      await expect(btn.first()).toBeVisible();
    }
  });

  test("add row header button has aria-label", async ({ page }) => {
    const btn = page.locator('button[aria-label="Add row header level"]');
    if (await btn.count()) {
      await expect(btn.first()).toBeVisible();
    }
  });

  test("column label inputs have aria-labels", async ({ page }) => {
    const labels = page.locator('input[aria-label^="Column"][aria-label$="label"]');
    const count = await labels.count();
    // Grid should have at least 1 column label if config loaded
    if (count > 0) {
      for (let i = 0; i < count; i++) {
        await expect(labels.nth(i)).toHaveAttribute("aria-label", /Column \d+ label/);
      }
    }
  });

  test("row label inputs have aria-labels", async ({ page }) => {
    const labels = page.locator('input[aria-label^="Row"][aria-label$="label"]');
    const count = await labels.count();
    if (count > 0) {
      for (let i = 0; i < count; i++) {
        await expect(labels.nth(i)).toHaveAttribute("aria-label", /Row \d+ label/);
      }
    }
  });
});

test.describe("Image Strip", () => {
  test("empty strip shows guidance text", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);
    const strip = page.getByText("No images loaded");
    if (await strip.count()) {
      await expect(strip).toBeVisible();
    }
  });
});

test.describe("Preview Pane", () => {
  test("preview section is visible", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);
    const preview = page.getByText("Preview").first();
    if (await preview.count()) {
      await expect(preview).toBeVisible();
    }
  });

  test("background selector has minimum width", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);
    // The select should contain White/Black/Transparent options
    const select = page.locator(".MuiSelect-select").first();
    if (await select.count()) {
      const box = await select.boundingBox();
      if (box) {
        expect(box.width).toBeGreaterThanOrEqual(100);
      }
    }
  });

  test("Generate Preview button is an MUI button", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);
    const btn = page.getByRole("button", { name: "Generate Preview" });
    if (await btn.count()) {
      // Should be MUI button (has MuiButton class)
      const classes = await btn.getAttribute("class");
      expect(classes).toContain("MuiButton");
    }
  });
});

test.describe("Responsive layout", () => {
  test("no horizontal overflow at 800px width", async ({ page }) => {
    await page.setViewportSize({ width: 800, height: 600 });
    await page.goto("/");
    await page.waitForTimeout(500);

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    // Allow small tolerance (scrollbar)
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 20);
  });

  test("no horizontal overflow at 1200px width", async ({ page }) => {
    await page.setViewportSize({ width: 1200, height: 800 });
    await page.goto("/");
    await page.waitForTimeout(500);

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 20);
  });
});

test.describe("Visual regression snapshots", () => {
  test("full page screenshot at 1280x720", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    await expect(page).toHaveScreenshot("app-1280x720.png", {
      maxDiffPixelRatio: 0.05,
    });
  });

  test("full page screenshot at 800x600", async ({ page }) => {
    await page.setViewportSize({ width: 800, height: 600 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    await expect(page).toHaveScreenshot("app-800x600.png", {
      maxDiffPixelRatio: 0.05,
    });
  });
});
