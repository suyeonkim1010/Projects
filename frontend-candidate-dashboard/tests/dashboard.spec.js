import { expect, test } from "@playwright/test";

test("filters candidates by search input", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("candidate-search").fill("Ava");

  await expect(page.getByTestId("candidate-card-3")).toBeVisible();
  await expect(page.getByTestId("candidate-card-1")).toHaveCount(0);
});

test("updates visible results when a status filter is selected", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("status-filter-interview").click();

  await expect(page.getByTestId("candidate-card-1")).toBeVisible();
  await expect(page.getByTestId("candidate-card-4")).toBeVisible();
  await expect(page.getByTestId("candidate-card-13")).toBeVisible();
  await expect(page.getByTestId("candidate-card-18")).toBeVisible();
  await expect(page.getByTestId("candidate-card-3")).toHaveCount(0);
  await expect(page.getByTestId("visible-candidates")).toHaveText("4");
});

test("filters candidates by region", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("region-filter-on").click();

  await expect(page.getByTestId("candidate-card-1")).toBeVisible();
  await expect(page.getByTestId("candidate-card-7")).toBeVisible();
  await expect(page.getByTestId("candidate-card-2")).toHaveCount(0);
});

test("updates the detail panel when a candidate card is clicked", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("candidate-card-3").click();

  await expect(page.getByTestId("candidate-detail-name")).toHaveText("Ava Singh");
  await expect(page.getByTestId("candidate-detail-panel")).toContainText("UI Developer");
});
