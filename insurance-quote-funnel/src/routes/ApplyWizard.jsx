import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { FormProvider as RhfProvider, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFormStore } from "../store/formStore.jsx";
import { stepSchemas } from "../store/formSchema.js";

import Step1 from "../steps/Step1.jsx";
import Step2 from "../steps/Step2.jsx";
import Step3 from "../steps/Step3.jsx";
import Step4 from "../steps/Step4.jsx";

const MIN = 1;
const MAX = 4;

function getFlatValues(data) {
  return {
    fullName: data.step1.fullName || "",
    email: data.step2.email || "",
    phone: data.step2.phone || "",
    experience: data.step3.experience || ""
  };
}

function canAccessStep(step, data) {
  if (step <= 1) return true;
  const values = getFlatValues(data);
  if (!stepSchemas[1].safeParse(values).success) return false;
  if (step <= 2) return true;
  if (!stepSchemas[2].safeParse(values).success) return false;
  if (step <= 3) return true;
  if (!stepSchemas[3].safeParse(values).success) return false;
  return true;
}

function getRedirectStep(step, data) {
  const values = getFlatValues(data);
  if (!stepSchemas[1].safeParse(values).success) return 1;
  if (step > 1 && !stepSchemas[2].safeParse(values).success) return 2;
  if (step > 2 && !stepSchemas[3].safeParse(values).success) return 3;
  return step;
}

export default function ApplyWizard() {
  const { step } = useParams();
  const navigate = useNavigate();
  const { isSubmissionLocked, data, reset, setData } = useFormStore();
  const [supportMode, setSupportMode] = useState(false);

  const currentStep = useMemo(() => Number(step || 1), [step]);

  const methods = useForm({
    resolver: zodResolver(stepSchemas[4]),
    mode: "onTouched",
    defaultValues: {
      fullName: data.step1.fullName || "",
      email: data.step2.email || "",
      phone: data.step2.phone || "",
      experience: data.step3.experience || ""
    }
  });

  const { watch } = methods;

  useEffect(() => {
    if (!Number.isInteger(currentStep) || currentStep < MIN || currentStep > MAX) {
      navigate("/apply/step/1", { replace: true });
    }
  }, [currentStep, navigate]);

  useEffect(() => {
    if (!canAccessStep(currentStep, data)) {
      const redirectStep = getRedirectStep(currentStep, data);
      navigate(`/apply/step/${redirectStep}`, { replace: true });
    }
  }, [currentStep, data, navigate]);

  useEffect(() => {
    if (isSubmissionLocked) {
      navigate("/apply/thank-you", { replace: true });
    }
  }, [isSubmissionLocked, navigate]);

  useEffect(() => {
    const subscription = watch((values) => {
      setData((prev) => ({
        ...prev,
        step1: { ...prev.step1, fullName: values.fullName || "" },
        step2: { ...prev.step2, email: values.email || "", phone: values.phone || "" },
        step3: { ...prev.step3, experience: values.experience || "" }
      }));
    });
    return () => subscription.unsubscribe();
  }, [setData, watch]);

  return (
    <div className="page">
      <header className="header">
        <div className="headerRow">
          <div>
            <h1>Multi-step Form</h1>
            <p className="muted">Step {currentStep} / {MAX}</p>
          </div>
          <div className="headerActions">
            <div className="savedText">
              {data.lastSavedAt
                ? `Saved ${new Date(data.lastSavedAt).toLocaleTimeString()}`
                : "Not saved yet"}
            </div>
            <button
              className="btnSecondary"
              type="button"
              onClick={() => setSupportMode((prev) => !prev)}
            >
              {supportMode ? "Hide support" : "Support mode"}
            </button>
            <button
              className="btnSecondary"
              type="button"
              onClick={() => {
                reset();
                navigate("/apply/step/1", { replace: true });
              }}
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      {supportMode && (
        <div className="supportPanel">
          <div className="supportTitle">Support mode</div>
          <div>Step: {currentStep}</div>
          <div>Last error: {data.lastError || "None"}</div>
          <pre className="supportPre">{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}

      <RhfProvider {...methods}>
        <div className="card">
          {currentStep === 1 && <Step1 />}
          {currentStep === 2 && <Step2 />}
          {currentStep === 3 && <Step3 />}
          {currentStep === 4 && <Step4 />}
        </div>
      </RhfProvider>
    </div>
  );
}
