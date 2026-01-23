import fs from "fs";
import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import ApplyWizard from "../routes/ApplyWizard.jsx";
import ThankYou from "../routes/ThankYou.jsx";
import { FormProvider } from "../store/formStore.jsx";

vi.mock("react-hot-toast", () => ({
  default: {
    error: vi.fn(),
    success: vi.fn()
  },
  Toaster: () => null
}));

import toast from "react-hot-toast";

const STORAGE_KEY = "apply_form_v1";

function renderWithProviders(initialEntries) {
  return render(
    <FormProvider>
      <MemoryRouter initialEntries={initialEntries}>
        <Routes>
          <Route path="/apply/step/:step" element={<ApplyWizard />} />
          <Route path="/apply/thank-you" element={<ThankYou />} />
        </Routes>
      </MemoryRouter>
    </FormProvider>
  );
}

beforeEach(() => {
  localStorage.clear();
  toast.error.mockClear();
  toast.success.mockClear();
});

describe("wizard flow", () => {
  it("validates Step 1 full name", async () => {
    renderWithProviders(["/apply/step/1"]);

    await userEvent.click(screen.getByRole("button", { name: /next/i }));

    expect(await screen.findByText("Please enter your full name.")).toBeInTheDocument();
    expect(toast.error).toHaveBeenCalledWith("Please fix the highlighted fields.");
  });

  it("includes responsive grid styles for Step 2", () => {
    const cssPath = path.resolve(process.cwd(), "src", "styles", "wizard.css");
    const css = fs.readFileSync(cssPath, "utf8");

    expect(css).toContain(".grid2 {");
    expect(css).toContain("@media (max-width: 768px)");
    expect(css).toContain(".grid2 { grid-template-columns: 1fr; }");
  });

  it("keeps URL step when prior data is complete", () => {
    const saved = {
      step1: { fullName: "Alex Lee" },
      step2: { email: "alex@example.com", phone: "5551234567" },
      step3: { experience: "This is enough text." },
      step4: { notes: "" },
      submitted: false,
      submittedAt: null,
      confirmationId: null,
      lastSavedAt: Date.now(),
      lastError: null
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));

    renderWithProviders(["/apply/step/3"]);

    expect(screen.getByText("Step 3: Experience")).toBeInTheDocument();
  });

  it("locks re-entry after submission", async () => {
    const saved = {
      step1: { fullName: "Alex Lee" },
      step2: { email: "alex@example.com", phone: "5551234567" },
      step3: { experience: "This is enough text." },
      step4: { notes: "" },
      submitted: true,
      submittedAt: Date.now(),
      confirmationId: "confirm-1",
      lastSavedAt: Date.now(),
      lastError: null
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));

    renderWithProviders(["/apply/step/2"]);

    expect(await screen.findByText("Thank you!")).toBeInTheDocument();
  });

  it("shows server error toast on submit failure", async () => {
    const saved = {
      step1: { fullName: "Alex Lee" },
      step2: { email: "alex@example.com", phone: "5551234567" },
      step3: { experience: "This is enough text." },
      step4: { notes: "" },
      submitted: false,
      submittedAt: null,
      confirmationId: null,
      lastSavedAt: Date.now(),
      lastError: null
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({})
    });

    renderWithProviders(["/apply/step/4"]);

    await userEvent.click(screen.getByRole("button", { name: /submit/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith("Server error. Try again.");
    });
  });
});
