import { useNavigate } from "react-router-dom";
import { useFormContext } from "react-hook-form";
import { useFormStore } from "../store/formStore.jsx";
import Field from "../ui/Field.jsx";
import toast from "react-hot-toast";

export default function Step1() {
  const navigate = useNavigate();
  const { setData } = useFormStore();
  const {
    register,
    trigger,
    formState: { errors }
  } = useFormContext();

  async function handleNext() {
    const ok = await trigger(["fullName"]);
    if (!ok) {
      setData((prev) => ({ ...prev, lastError: "Step 1 validation failed." }));
      toast.error("Please fix the highlighted fields.");
      return;
    }
    setData((prev) => ({ ...prev, lastError: null }));
    navigate("/apply/step/2");
  }

  return (
    <div>
      <h2 className="title">Step 1: Basic</h2>

      <Field
        label="Full name"
        name="fullName"
        register={register}
        error={errors.fullName?.message}
        placeholder="Suyeon Kim"
      />

      <div className="row">
        <button className="btn" onClick={handleNext}>
          Next
        </button>
      </div>
    </div>
  );
}
