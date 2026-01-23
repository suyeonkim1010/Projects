import { Link, Navigate, useLocation } from "react-router-dom";
import { useFormStore } from "../../app/formStore";

export default function ThankYouPage() {
  const location = useLocation();
  const { data } = useFormStore();
  const confirmationId =
    (location.state as { confirmationId?: string } | null)?.confirmationId ||
    data.confirmationId;

  if (!data.submitted) {
    return <Navigate to="/apply/step/1" replace />;
  }

  return (
    <div>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Thanks!</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        We received your request. We’ll contact you soon.
      </p>

      <div style={{ marginTop: 16, color: "#555" }}>
        Confirmation: {confirmationId ?? "Pending"}
      </div>

      <div style={{ marginTop: 24 }}>
        <Link to="/apply/step/1">Back to quote</Link>
      </div>
    </div>
  );
}
