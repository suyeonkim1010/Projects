import { useFormContext } from "react-hook-form";
import type { QuoteFormData } from "../quoteSchema";

export default function Step2Vehicle() {
  const {
    register,
    formState: { errors }
  } = useFormContext<QuoteFormData>();

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
        <label>
          Year
          <input {...register("vehicleYear")} placeholder="2020" style={inputStyle} />
          {errors.vehicleYear && <div style={errStyle}>{errors.vehicleYear.message}</div>}
        </label>

        <label>
          Make
          <input {...register("vehicleMake")} placeholder="Toyota" style={inputStyle} />
          {errors.vehicleMake && <div style={errStyle}>{errors.vehicleMake.message}</div>}
        </label>
      </div>

      <label>
        Model
        <input {...register("vehicleModel")} placeholder="Corolla" style={inputStyle} />
        {errors.vehicleModel && <div style={errStyle}>{errors.vehicleModel.message}</div>}
      </label>

      <div style={{ fontSize: 13, color: "#666" }}>
        Tip: In production, Step 2 often includes VIN too, but this is simplified.
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  display: "block",
  width: "100%",
  marginTop: 6,
  padding: "10px 12px",
  borderRadius: 10,
  border: "1px solid #ccc",
  outline: "none"
};

const errStyle: React.CSSProperties = {
  marginTop: 6,
  color: "#b00020",
  fontSize: 13
};
