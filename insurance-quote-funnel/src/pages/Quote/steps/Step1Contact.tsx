import { useFormContext } from "react-hook-form";
import type { QuoteFormData } from "../quoteSchema";

export default function Step1Contact() {
  const {
    register,
    formState: { errors }
  } = useFormContext<QuoteFormData>();

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <label>
        Full name
        <input
          {...register("fullName")}
          placeholder="Suyeon Kim"
          style={inputStyle}
          autoComplete="name"
        />
        {errors.fullName && <div style={errStyle}>{errors.fullName.message}</div>}
      </label>

      <label>
        Email
        <input
          {...register("email")}
          placeholder="you@email.com"
          style={inputStyle}
          autoComplete="email"
        />
        {errors.email && <div style={errStyle}>{errors.email.message}</div>}
      </label>

      <label>
        Phone
        <input
          {...register("phone")}
          placeholder="780-123-4567"
          style={inputStyle}
          autoComplete="tel"
        />
        {errors.phone && <div style={errStyle}>{errors.phone.message}</div>}
      </label>
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
