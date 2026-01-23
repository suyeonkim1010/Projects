import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { FormProvider, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import ProgressBar from "../../components/ProgressBar";
import Step1Contact from "./steps/Step1Contact";
import Step2Vehicle from "./steps/Step2Vehicle";
import Step3Coverage from "./steps/Step3Coverage";
import { quoteSchema, type QuoteFormData } from "./quoteSchema";
import { useFormStore } from "../../app/formStore";

export default function QuotePage() {
  const navigate = useNavigate();
  const { step } = useParams();
  const { data, setData } = useFormStore();

  const [serverError, setServerError] = useState<string | null>(null);

  const currentStep = useMemo(() => Number(step || 1), [step]);

  const methods = useForm<QuoteFormData>({
    resolver: zodResolver(quoteSchema),
    mode: "onTouched",
    defaultValues: {
      fullName: data.values.fullName ?? "",
      email: data.values.email ?? "",
      phone: data.values.phone ?? "",
      vehicleYear: data.values.vehicleYear ?? "",
      vehicleMake: data.values.vehicleMake ?? "",
      vehicleModel: data.values.vehicleModel ?? "",
      coverageType: data.values.coverageType,
      hasAccidents: data.values.hasAccidents
    }
  });

  const { handleSubmit, trigger, formState, getValues, watch } = methods;

  const stepLabels = ["Contact", "Vehicle", "Coverage"];
  const totalSteps = 3;

  const stepFields = useMemo(() => {
    return {
      1: ["fullName", "email", "phone"] as const,
      2: ["vehicleYear", "vehicleMake", "vehicleModel"] as const,
      3: ["coverageType", "hasAccidents"] as const
    };
  }, []);

  useEffect(() => {
    if (!Number.isInteger(currentStep) || currentStep < 1 || currentStep > totalSteps) {
      navigate("/apply/step/1", { replace: true });
    }
  }, [currentStep, navigate, totalSteps]);

  useEffect(() => {
    if (data.submitted) {
      navigate("/apply/thank-you", { replace: true });
    }
  }, [data.submitted, navigate]);

  useEffect(() => {
    const subscription = watch((values) => {
      setData((prev) => ({ ...prev, values }));
    });
    return () => subscription.unsubscribe();
  }, [setData, watch]);

  async function goNext() {
    setServerError(null);

    const fields = stepFields[currentStep as 1 | 2 | 3];
    const ok = await trigger(fields as any); // validate ONLY current step fields
    if (!ok) return;

    const nextStep = currentStep === 3 ? 3 : currentStep + 1;
    navigate(`/apply/step/${nextStep}`);
  }

  function goBack() {
    setServerError(null);
    const prevStep = currentStep === 1 ? 1 : currentStep - 1;
    navigate(`/apply/step/${prevStep}`);
  }

  async function onSubmit(data: QuoteFormData) {
    setServerError(null);

    try {
      // In production, this would be an API call (POST /quotes).
      // Simulate success with a small delay.
      await new Promise((r) => setTimeout(r, 600));

      const confirmationId = crypto.randomUUID();
      setData((prev) => ({
        ...prev,
        submitted: true,
        submittedAt: Date.now(),
        confirmationId,
        values: data
      }));

      // Success -> Thank you page
      navigate("/apply/thank-you", { replace: true, state: { confirmationId } });
    } catch (e) {
      setServerError("Something went wrong while submitting. Please try again.");
    }
  }

  return (
    <div>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Auto Insurance Quote</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        Multi-step funnel form (responsive + validation + error handling).
      </p>

      <ProgressBar current={currentStep} total={totalSteps} labels={stepLabels} />

      <div style={cardStyle}>
        {serverError && <div style={serverErrStyle}>{serverError}</div>}

        <FormProvider {...methods}>
          <form onSubmit={handleSubmit(onSubmit)}>
            {currentStep === 1 && <Step1Contact />}
            {currentStep === 2 && <Step2Vehicle />}
            {currentStep === 3 && <Step3Coverage />}

            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <button
                type="button"
                onClick={goBack}
                disabled={currentStep === 1 || formState.isSubmitting}
                style={btnSecondary}
              >
                Back
              </button>

              <div style={{ flex: 1 }} />

              {currentStep < 3 ? (
                <button
                  type="button"
                  onClick={goNext}
                  disabled={formState.isSubmitting}
                  style={btnPrimary}
                >
                  Next
                </button>
              ) : (
                <button type="submit" disabled={formState.isSubmitting} style={btnPrimary}>
                  {formState.isSubmitting ? "Submitting..." : "Submit"}
                </button>
              )}
            </div>

            <details style={{ marginTop: 18, color: "#666" }}>
              <summary>Debug: current values</summary>
              <pre style={{ whiteSpace: "pre-wrap" }}>
                {JSON.stringify(getValues(), null, 2)}
              </pre>
            </details>
          </form>
        </FormProvider>
      </div>
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  padding: 16,
  border: "1px solid #ddd",
  borderRadius: 14
};

const serverErrStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderRadius: 12,
  border: "1px solid #b00020",
  color: "#b00020",
  background: "#fff5f6",
  marginBottom: 14
};

const btnPrimary: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: 12,
  border: "1px solid #111",
  background: "#111",
  color: "#fff",
  cursor: "pointer",
  minWidth: 110
};

const btnSecondary: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: 12,
  border: "1px solid #ccc",
  background: "#fff",
  color: "#111",
  cursor: "pointer",
  minWidth: 110
};
