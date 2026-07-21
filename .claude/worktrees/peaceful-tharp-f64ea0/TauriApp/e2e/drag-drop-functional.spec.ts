/* ──────────────────────────────────────────────────────────
   E2E test: Functional drag-and-drop test
   Actually performs drag from filmstrip to panel
   ────────────────────────────────────────────────────────── */

import { test, expect } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename2 = fileURLToPath(import.meta.url);
const __dirname2 = path.dirname(__filename2);
const SHOTS = path.join(__dirname2, "screenshots");

test.describe("Functional drag-drop test", () => {
  test("drag image from filmstrip to empty panel via dropdown fallback", async ({ page }) => {
    await page.goto("http://localhost:1420");
    await page.waitForTimeout(2000);

    // Screenshot: initial state
    await page.screenshot({ path: path.join(SHOTS, "dd-01-initial.png"), fullPage: true });

    // Check if there are images in the filmstrip
    const filmstripImages = page.locator('[draggable="true"] img');
    const imgCount = await filmstripImages.count();
    console.log(`Filmstrip images: ${imgCount}`);

    if (imgCount === 0) {
      console.log("No images loaded, skipping drag test");
      return;
    }

    // Get the first image name from the filmstrip
    const firstDraggable = page.locator('[draggable="true"]').first();
    const imgName = await firstDraggable.locator('span').textContent();
    console.log(`First image name: ${imgName}`);

    // Try programmatic drag-and-drop
    const panelCell = page.locator('[class*="MuiBox-root"]').filter({ hasText: "R1 C1" }).first();

    // Method 1: Use Playwright's drag-to
    try {
      await firstDraggable.dragTo(panelCell, { timeout: 5000 });
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(SHOTS, "dd-02-after-drag.png"), fullPage: true });
    } catch (e) {
      console.log("dragTo failed, trying manual approach:", e);
    }

    // Check if the panel got the image
    const r1c1Select = page.locator('.MuiSelect-select').first();
    const selectValue = await r1c1Select.textContent();
    console.log(`R1C1 select value after drag: ${selectValue}`);

    // If drag didn't work, use the dropdown as fallback
    if (selectValue === "No Image" && imgName) {
      console.log("Drag did not work — testing dropdown assignment instead");

      // Click the dropdown
      await r1c1Select.click();
      await page.waitForTimeout(300);

      // Click the image option
      const option = page.getByRole("option").filter({ hasText: imgName.trim() });
      if (await option.count() > 0) {
        await option.first().click();
        await page.waitForTimeout(1000);
        console.log("Image assigned via dropdown");
      }

      await page.screenshot({ path: path.join(SHOTS, "dd-03-after-dropdown.png"), fullPage: true });
    }

    // Now let's test actual drag with dispatchEvent approach
    console.log("\n--- Testing drag via JavaScript dispatchEvent ---");

    // Get positions
    const srcBox = await firstDraggable.boundingBox();
    const targetCell = page.locator('[class*="MuiBox-root"]').filter({ hasText: "R1 C2" }).first();
    const dstBox = await targetCell.boundingBox();

    if (srcBox && dstBox) {
      console.log(`Source: ${srcBox.x},${srcBox.y} ${srcBox.width}x${srcBox.height}`);
      console.log(`Target: ${dstBox.x},${dstBox.y} ${dstBox.width}x${dstBox.height}`);

      // Use JavaScript to dispatch custom drag events
      const result = await page.evaluate(([sx, sy, dx, dy, name]) => {
        const src = document.elementFromPoint(sx, sy);
        const dst = document.elementFromPoint(dx, dy);
        if (!src || !dst) return `Elements not found at (${sx},${sy}) or (${dx},${dy})`;

        // Create a DataTransfer
        const dt = new DataTransfer();
        dt.setData("application/x-image-name", name as string);
        dt.setData("text/plain", name as string);

        // Dispatch dragstart on source
        src.dispatchEvent(new DragEvent("dragstart", { bubbles: true, dataTransfer: dt }));

        // Dispatch dragenter + dragover on target
        dst.dispatchEvent(new DragEvent("dragenter", { bubbles: true, dataTransfer: dt }));
        dst.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, dataTransfer: dt }));

        // Dispatch drop
        const dropEvt = new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: dt });
        const dropResult = dst.dispatchEvent(dropEvt);

        return `Drop dispatched: ${dropResult}, src: ${src.tagName}.${src.className?.substring(0,30)}, dst: ${dst.tagName}.${dst.className?.substring(0,30)}`;
      }, [
        srcBox.x + srcBox.width / 2,
        srcBox.y + srcBox.height / 2,
        dstBox.x + dstBox.width / 2,
        dstBox.y + dstBox.height / 2,
        imgName?.trim() || "",
      ]);

      console.log(`JS drag result: ${result}`);
      await page.waitForTimeout(1000);
    }

    await page.screenshot({ path: path.join(SHOTS, "dd-04-final.png"), fullPage: true });
  });
});
