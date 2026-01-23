import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useFormContext } from "react-hook-form";
import { useFormStore } from "../store/formStore.jsx";
import toast from "react-hot-toast";

async function submitApplication(payload, mode) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  const query = mode ? `?mode=${mode}` : "";

  try {
    const res = await fetch(`http://localhost:4000/api/applications${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    if (!res.ok) {
      const errorBody = await res.json().catch(() => ({}));
      const message = errorBody.error || `HTTP_${res.status}`;
      throw new Error(message);
    }

    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("TIMEOUT");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export default function Step4() {
  const navigate = useNavigate();
  const { data, setData } = useFormStore();
  const { trigger, getValues, setError } = useFormContext();
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState(null);
  const lastPayloadRef = useRef(null);

  async function submitPayload(payload) {
    try {
      setSubmitting(true);
      setServerError(null);
      const mode = new URLSearchParams(window.location.search).get("mode");
      const res = await submitApplication(payload, mode);
      const confirmationId = res.confirmationId || crypto.randomUUID();

      setData((prev) => ({
        ...prev,
        submitted: true,
        submittedAt: Date.now(),
        confirmationId,
        lastError: null
      }));

      toast.success("Submitted!");
      navigate("/apply/thank-you", {
        replace: true,
        state: { confirmationId }
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "UNKNOWN";
      setData((prev) => ({ ...prev, lastError: message }));

      if (err && typeof err === "object" && err.type === "validation") {
        const fieldErrors = err.fieldErrors || {};
        Object.entries(fieldErrors).forEach(([field, errorMessage]) => {
          setError(field, { type: "server", message: errorMessage });
        });
        toast.error("Some fields need your attention.");
        return;
      }

      setServerError({
        title: "We could not submit your application",
        message:
          message === "TIMEOUT"
            ? "Network issue detected. Please check your connection and try again."
            : "Something went wrong on our side. Please try again.",
        canRetry: true,
        debugId: message
      });

      if (message === "TIMEOUT") {
        toast.error("Request timed out. Please try again.");
      } else if (message.startsWith("HTTP_")) {
        toast.error("Server error. Try again.");
      } else {
        toast.error("Submit failed. Try again.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubmit() {
    const ok = await trigger();
    if (!ok) {
      setData((prev) => ({ ...prev, lastError: "Final validation failed." }));
      toast.error("Please fix the highlighted fields.");
      return;
    }

    try {
      if (submitting) return;
      const values = getValues();
      const payload = {
        step1: { fullName: values.fullName },
        step2: { email: values.email, phone: values.phone },
        step3: { experience: values.experience },
        step4: { notes: data.step4.notes }
      };
      lastPayloadRef.current = payload;
      await submitPayload(payload);
    } catch {}
  }

  async function handleRetry() {
    if (!lastPayloadRef.current || submitting) return;
    await submitPayload(lastPayloadRef.current);
  }

  return (
    <div>
      <h2 className="title">Step 4: Review and Submit</h2>

      {serverError ? (
        <div className="errorBanner" role="alert">
          <div>
            <strong>{serverError.title}</strong>
            <div>{serverError.message}</div>
            {serverError.debugId ? (
              <div className="errorMeta">Ref: {serverError.debugId}</div>
            ) : null}
          </div>
          <div className="bannerActions">
            {serverError.canRetry ? (
              <button
                className="btnSecondary"
                type="button"
                onClick={handleRetry}
                disabled={submitting}
              >
                Retry
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="summary">
        <div><b>Name:</b> {data.step1.fullName || "-"}</div>
        <div><b>Email:</b> {data.step2.email || "-"}</div>
        <div><b>Phone:</b> {data.step2.phone || "-"}</div>
      </div>

      <div className="row spaceBetween">
        <button className="btnSecondary" onClick={() => navigate("/apply/step/3")}>
          Back
        </button>
        <button
          className={`btn ${submitting ? "btnLoading" : ""}`}
          onClick={handleSubmit}
          disabled={submitting}
        >
          {submitting ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Submitting
            </>
          ) : (
            "Submit"
          )}
        </button>
      </div>
    </div>
  );
}
