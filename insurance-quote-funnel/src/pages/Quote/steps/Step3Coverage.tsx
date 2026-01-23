import { useFormContext } from "react-hook-form";
import type { QuoteFormData } from "../quoteSchema";

export default function Step3Coverage() {
  const {
    register,
    formState: { errors },
    watch
  } = useFormContext<QuoteFormData>();

  const coverage = watch("coverageType");

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Coverage type</div>

        <div style={{ display: "grid", gap: 10 }}>
          <label style={radioCard(coverage === "basic")}>
            <input type="radio" value="basic" {...register("coverageType")} />
            <span style={{ marginLeft: 10 }}>
              <b>Basic</b> - cheapest, minimal coverage
            </span>
          </label>

          <label style={radioCard(coverage === "standard")}>
            <input type="radio" value="standard" {...register("coverageType")} />
            <span style={{ marginLeft: 10 }}>
              <b>Standard</b> - balanced option
            </span>
          </label>

          <label style={radioCard(coverage === "premium")}>
            <input type="radio" value="premium" {...register("coverageType")} />
            <span style={{ marginLeft: 10 }}>
              <b>Premium</b> - best coverage + add-ons
            </span>
          </label>
        </div>

        {errors.coverageType && <div style={errStyle}>{errors.coverageType.message}</div>}
      </div>

      <div>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Any accidents in last 3 years?</div>
        <div style={{ display: "flex", gap: 16 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="radio" value="yes" {...register("hasAccidents")} />
            Yes
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="radio" value="no" {...register("hasAccidents")} />
            No
          </label>
        </div>
        {errors.hasAccidents && <div style={errStyle}>{errors.hasAccidents.message}</div>}
      </div>

      <div style={{ fontSize: 13, color: "#666" }}>
        In production, this step would include driver count, address, and consent.
      </div>
    </div>
  );
}

const radioCard = (active: boolean): React.CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "12px 12px",
  borderRadius: 12,
  border: `1px solid ${active ? "#111" : "#ddd"}`,
  background: active ? "#f6f6f6" : "#fff",
  cursor: "pointer"
});

const errStyle: React.CSSProperties = {
  marginTop: 8,
  color: "#b00020",
  fontSize: 13
};
