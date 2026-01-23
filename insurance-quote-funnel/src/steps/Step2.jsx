import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useFormContext } from "react-hook-form";
import { useFormStore } from "../store/formStore.jsx";
import Field from "../ui/Field.jsx";
import toast from "react-hot-toast";

async function saveStep2ToServer() {
  await new Promise((r) => setTimeout(r, 400));
}

export default function Step2() {
  const navigate = useNavigate();
  const { setData } = useFormStore();
  const {
    register,
    trigger,
    formState: { errors }
  } = useFormContext();

  const [serverError, setServerError] = useState("");

  async function handleNext() {
    setServerError("");
    const ok = await trigger(["email", "phone"]);
    if (!ok) {
      setData((prev) => ({ ...prev, lastError: "Step 2 validation failed." }));
      toast.error("Please fix the highlighted fields.");
      return;
    }

    try {
      await saveStep2ToServer();
      setData((prev) => ({ ...prev, lastError: null }));
      navigate("/apply/step/3");
    } catch (err) {
      setServerError("Server error while saving. Please try again.");
      setData((prev) => ({ ...prev, lastError: "Step 2 server error." }));
      toast.error("Server error. Try again.");
    }
  }

  return (
    <div>
      <h2 className="title">Step 2: Contact</h2>

      {serverError ? (
        <div className="serverError" role="alert">
          {serverError}
        </div>
      ) : null}

      <div className="grid2">
        <Field
          label="Email"
          name="email"
          register={register}
          error={errors.email?.message}
          placeholder="you@example.com"
        />

        <Field
          label="Phone"
          name="phone"
          register={register}
          error={errors.phone?.message}
          placeholder="(555) 123-4567"
        />
      </div>

      <div className="row spaceBetween">
        <button className="btnSecondary" onClick={() => navigate("/apply/step/1")}>
          Back
        </button>
        <button className="btn" onClick={handleNext}>
          Next
        </button>
      </div>
    </div>
  );
}
