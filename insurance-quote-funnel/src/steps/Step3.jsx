import { useNavigate } from "react-router-dom";
import { useFormContext } from "react-hook-form";
import { useFormStore } from "../store/formStore.jsx";
import toast from "react-hot-toast";

export default function Step3() {
  const navigate = useNavigate();
  const { setData } = useFormStore();
  const {
    register,
    trigger,
    formState: { errors }
  } = useFormContext();

  const errorId = errors.experience ? "experience-error" : undefined;

  async function handleNext() {
    const ok = await trigger(["experience"]);
    if (!ok) {
      setData((prev) => ({ ...prev, lastError: "Step 3 validation failed." }));
      toast.error("Please fix the highlighted fields.");
      return;
    }
    setData((prev) => ({ ...prev, lastError: null }));
    navigate("/apply/step/4");
  }

  return (
    <div>
      <h2 className="title">Step 3: Experience</h2>

      <div className="field">
        <label className="label" htmlFor="experience">Short summary</label>
        <textarea
          id="experience"
          className={`textarea ${errors.experience ? "inputError" : ""}`}
          {...register("experience")}
          placeholder="Briefly describe relevant experience..."
          aria-invalid={Boolean(errors.experience)}
          aria-describedby={errorId}
        />
        {errors.experience ? (
          <div className="fieldError" id={errorId} role="alert">
            {errors.experience.message}
          </div>
        ) : null}
      </div>

      <div className="row spaceBetween">
        <button className="btnSecondary" onClick={() => navigate("/apply/step/2")}>
          Back
        </button>
        <button className="btn" onClick={handleNext}>
          Next
        </button>
      </div>
    </div>
  );
}
