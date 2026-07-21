/* ──────────────────────────────────────────────────────────
   E2E tests for EditPanelDialog — validates that the panel
   settings dialog opens, tabs work, and UI fixes are intact.
   Runs against the Vite dev server (no Tauri/Python needed).
   ────────────────────────────────────────────────────────── */

import { test, expect } from "@playwright/test";

/* ── Helper: open the edit dialog for the first panel cell ── */
async function openEditDialog(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.waitForTimeout(1200);

  // The edit button is hidden (opacity 0) by default — hover the first panel cell
  const cell = page.locator("text=R1 C1").first();
  await cell.hover();
  await page.waitForTimeout(300);

  // Click the edit button (aria-label="Edit panel")
  const editBtn = page.locator('button[aria-label="Edit panel"]').first();
  // Force-click since it may be visually hidden (opacity: 0) but in DOM
  await editBtn.click({ force: true });
  await page.waitForTimeout(800);

  // Verify dialog actually opened
  await expect(page.locator("text=Edit Panel R1 C1")).toBeVisible({ timeout: 5000 });
}

test.describe("EditPanelDialog — Opening & Closing", () => {
  test("edit dialog opens with correct title", async ({ page }) => {
    await openEditDialog(page);
    const title = page.locator("text=Edit Panel R1 C1");
    await expect(title).toBeVisible();
  });

  test("edit dialog closes on Cancel button", async ({ page }) => {
    await openEditDialog(page);
    const cancelBtn = page.getByRole("button", { name: "Cancel" });
    await cancelBtn.click();
    await page.waitForTimeout(300);
    const title = page.locator("text=Edit Panel R1 C1");
    await expect(title).not.toBeVisible();
  });
});

test.describe("EditPanelDialog — Tab Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
  });

  test("all 6 tabs are visible", async ({ page }) => {
    const tabLabels = [
      "Crop/Resize",
      "Adjustments",
      "Labels",
      "Scale Bar",
      "Symbols",
      "Zoom Inset",
    ];
    for (const label of tabLabels) {
      const tab = page.getByRole("tab", { name: label });
      await expect(tab).toBeVisible();
    }
  });

  test("clicking each tab shows its content", async ({ page }) => {
    // Tab 0: Crop/Resize — should show "Rotation" text
    await page.getByRole("tab", { name: "Crop/Resize" }).click();
    await page.waitForTimeout(200);
    await expect(page.getByText("Rotation").first()).toBeVisible();

    // Tab 1: Image Adjustments — should show "Brightness" text
    await page.getByRole("tab", { name: "Adjustments" }).click();
    await page.waitForTimeout(200);
    await expect(page.getByText("Brightness").first()).toBeVisible();

    // Tab 2: Labels — should show "Add Label" button
    await page.getByRole("tab", { name: "Labels" }).click();
    await page.waitForTimeout(200);
    await expect(
      page.getByRole("button", { name: "Add Label" })
    ).toBeVisible();

    // Tab 3: Scale Bar — should show "Enable Scale Bar" switch
    await page.getByRole("tab", { name: "Scale Bar" }).click();
    await page.waitForTimeout(200);
    await expect(page.getByText("Enable Scale Bar")).toBeVisible();

    // Tab 4: Symbols — should show "Add Symbol" button
    await page.getByRole("tab", { name: "Symbols" }).click();
    await page.waitForTimeout(200);
    await expect(
      page.getByRole("button", { name: "Add Symbol" })
    ).toBeVisible();

    // Tab 5: Zoom Inset — should show "Enable Zoom Inset" switch
    await page.getByRole("tab", { name: "Zoom Inset" }).click();
    await page.waitForTimeout(200);
    await expect(page.getByText("Enable Zoom Inset")).toBeVisible();
  });
});

test.describe("EditPanelDialog — Crop/Resize Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Crop/Resize" }).click();
    await page.waitForTimeout(200);
  });

  test("rotation slider is present", async ({ page }) => {
    const slider = page.locator(".MuiSlider-root").first();
    await expect(slider).toBeVisible();
  });

  test("aspect ratio toggle group wraps properly", async ({ page }) => {
    // The ToggleButtonGroup should have flexWrap: wrap
    const tbg = page.locator(".MuiToggleButtonGroup-root").first();
    if (await tbg.count()) {
      const box = await tbg.boundingBox();
      expect(box).not.toBeNull();
      if (box) {
        // Should not overflow dialog width (dialog is ~340px left panel)
        expect(box.width).toBeLessThanOrEqual(350);
      }
    }
  });
});

test.describe("EditPanelDialog — Image Adjustments Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Adjustments" }).click();
    await page.waitForTimeout(200);
  });

  test("brightness/contrast/saturation sliders are present", async ({
    page,
  }) => {
    await expect(page.getByText("Brightness").first()).toBeVisible();
    await expect(page.getByText("Contrast").first()).toBeVisible();
    await expect(page.getByText("Saturation").first()).toBeVisible();
  });

  test("tone curve section is present", async ({ page }) => {
    await expect(page.getByText("Tone Curve").first()).toBeVisible();
    await expect(page.getByText("Highlights").first()).toBeVisible();
    await expect(page.getByText("Shadows").first()).toBeVisible();
  });

  test("effects checkboxes are present", async ({ page }) => {
    await expect(page.getByText("Effects").first()).toBeVisible();
    await expect(page.getByText("Invert").first()).toBeVisible();
    await expect(page.getByText("Grayscale").first()).toBeVisible();
  });
});

test.describe("EditPanelDialog — Labels Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Labels" }).click();
    await page.waitForTimeout(200);
  });

  test("Add Label button adds a label entry", async ({ page }) => {
    const addBtn = page.getByRole("button", { name: "Add Label" });
    await addBtn.click();
    await page.waitForTimeout(300);

    // Should show "Label 1" text
    await expect(page.getByText("Label 1").first()).toBeVisible();

    // Should have a text input field
    const textField = page.locator('input[type="text"]').first();
    if (await textField.count()) {
      await expect(textField).toBeVisible();
    }
  });

  test("label delete button has aria-label", async ({ page }) => {
    // Add a label first
    await page.getByRole("button", { name: "Add Label" }).click();
    await page.waitForTimeout(300);

    const deleteBtn = page.locator(
      'button[aria-label="Remove label 1"]'
    );
    await expect(deleteBtn).toBeVisible();
  });

  test("label ListItem has padding for delete button", async ({ page }) => {
    await page.getByRole("button", { name: "Add Label" }).click();
    await page.waitForTimeout(300);

    // The ListItem should have pr: 5 (40px) so delete button doesn't overlap
    const listItem = page.locator(".MuiListItem-root").first();
    if (await listItem.count()) {
      const box = await listItem.boundingBox();
      const deleteBtn = page.locator(
        'button[aria-label="Remove label 1"]'
      );
      const deleteBtnBox = await deleteBtn.boundingBox();

      // Delete button should be within the list item bounds
      if (box && deleteBtnBox) {
        expect(deleteBtnBox.x + deleteBtnBox.width).toBeLessThanOrEqual(
          box.x + box.width + 5
        );
      }
    }
  });
});

test.describe("EditPanelDialog — Scale Bar Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Scale Bar" }).click();
    await page.waitForTimeout(200);
  });

  test("enable scale bar switch works", async ({ page }) => {
    const switchEl = page.locator(".MuiSwitch-root").first();
    await expect(switchEl).toBeVisible();

    // Click to enable
    await switchEl.click();
    await page.waitForTimeout(300);

    // After enabling, scale bar controls should appear
    await expect(page.getByText("Measurement").first()).toBeVisible();
  });

  test("scale bar preview does not overflow its container", async ({ page }) => {
    // Enable scale bar — click the switch input for reliability
    const switchInput = page.locator(".MuiSwitch-root input[type='checkbox']").first();
    await switchInput.check({ force: true });
    await page.waitForTimeout(800);

    // After enabling, the "Measurement" heading appears — the preview is above it
    await expect(page.getByText("Measurement").first()).toBeVisible({ timeout: 10000 });

    // The left controls panel is 340px wide; check the scale bar controls
    // don't spill outside it. Find the controls panel (first child of the flex container).
    const controlsPanel = page.locator(".MuiDialogContent-root > div > div").first();
    if (await controlsPanel.count()) {
      const box = await controlsPanel.boundingBox();
      if (box) {
        // Controls panel should be ≤ 420px (it's set to 400px width)
        expect(box.width).toBeLessThanOrEqual(420);
      }
    }
  });
});

test.describe("EditPanelDialog — Symbols Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Symbols" }).click();
    await page.waitForTimeout(200);
  });

  test("Add Symbol button adds a symbol entry", async ({ page }) => {
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(300);

    await expect(page.getByText("Symbol 1").first()).toBeVisible();
  });

  test("symbol delete button has aria-label", async ({ page }) => {
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(300);

    const deleteBtn = page.locator(
      'button[aria-label="Remove symbol 1"]'
    );
    await expect(deleteBtn).toBeVisible();
  });

  test("symbol shape toggle group wraps properly", async ({ page }) => {
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(300);

    const tbg = page.locator(".MuiToggleButtonGroup-root").first();
    if (await tbg.count()) {
      const box = await tbg.boundingBox();
      expect(box).not.toBeNull();
      if (box) {
        // Should not overflow the controls panel
        expect(box.width).toBeLessThanOrEqual(350);
      }
    }
  });

  test("symbol ListItem has padding for delete button", async ({ page }) => {
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(300);

    const listItem = page.locator(".MuiListItem-root").first();
    if (await listItem.count()) {
      const box = await listItem.boundingBox();
      const deleteBtn = page.locator(
        'button[aria-label="Remove symbol 1"]'
      );
      const deleteBtnBox = await deleteBtn.boundingBox();

      if (box && deleteBtnBox) {
        expect(deleteBtnBox.x + deleteBtnBox.width).toBeLessThanOrEqual(
          box.x + box.width + 5
        );
      }
    }
  });

  test("all 5 shape buttons are present", async ({ page }) => {
    await page.getByRole("button", { name: "Add Symbol" }).click();
    await page.waitForTimeout(300);

    const shapes = ["Arrow", "Star", "Rectangle", "Ellipse", "Cross"];
    for (const shape of shapes) {
      const btn = page.locator(`button[title="${shape}"]`);
      await expect(btn.first()).toBeVisible();
    }
  });
});

test.describe("EditPanelDialog — Zoom Inset Tab", () => {
  test.beforeEach(async ({ page }) => {
    await openEditDialog(page);
    await page.getByRole("tab", { name: "Zoom Inset" }).click();
    await page.waitForTimeout(200);
  });

  test("enable zoom inset switch works", async ({ page }) => {
    const switchEl = page.locator(".MuiSwitch-root").first();
    await expect(switchEl).toBeVisible();

    await switchEl.click();
    await page.waitForTimeout(300);

    // After enabling, zoom controls should appear
    await expect(page.getByText("Zoom Area").first()).toBeVisible();
  });

  test("inset type selector appears after enabling", async ({ page }) => {
    const switchInput = page.locator(".MuiSwitch-root input[type='checkbox']").first();
    await switchInput.check({ force: true });
    await page.waitForTimeout(800);

    // The Select should show "Standard Zoom" as default
    await expect(page.getByText("Standard Zoom").first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe("EditPanelDialog — Responsive behavior", () => {
  test("dialog renders without overflow at 800px viewport", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 800, height: 600 });
    await openEditDialog(page);

    // Dialog should be visible
    const dialog = page.locator(".MuiDialog-root").first();
    await expect(dialog).toBeVisible();

    // Dialog content should not overflow viewport
    const dialogPaper = page.locator(".MuiDialog-paper").first();
    const box = await dialogPaper.boundingBox();
    if (box) {
      expect(box.x).toBeGreaterThanOrEqual(-5);
      expect(box.x + box.width).toBeLessThanOrEqual(810);
    }
  });

  test("dialog renders without overflow at 1280px viewport", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await openEditDialog(page);

    const dialogPaper = page.locator(".MuiDialog-paper").first();
    const box = await dialogPaper.boundingBox();
    if (box) {
      expect(box.x).toBeGreaterThanOrEqual(-5);
      expect(box.x + box.width).toBeLessThanOrEqual(1290);
    }
  });
});
