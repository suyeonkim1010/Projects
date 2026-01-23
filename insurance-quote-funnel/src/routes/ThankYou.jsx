import { useLocation, Navigate, useNavigate } from "react-router-dom";
import { useFormStore } from "../store/formStore.jsx";

export default function ThankYou() {
  const location = useLocation();
  const navigate = useNavigate();
  const { data, isSubmissionLocked, reset } = useFormStore();

  if (!data.submitted) {
    return <Navigate to="/apply/step/1" replace />;
  }

  const confirmationId = location.state?.confirmationId || data.confirmationId;

  return (
    <div className="page">
      <header className="header">
        <h1>Thank you!</h1>
        <p className="muted">
          {confirmationId ? `Confirmation: ${confirmationId}` : "Submitted successfully."}
        </p>
      </header>

      <div className="card">
        <p>
          {isSubmissionLocked
            ? "Your submission is locked for a while to prevent duplicate submissions."
            : "Lock expired. You can submit again if needed."}
        </p>

        <div className="ctaGrid">
          <div className="ctaCard">
            <div>
              <strong>Book a quick call</strong>
              <div className="muted">Schedule a 15-minute consult.</div>
            </div>
            <button className="btn">Schedule request</button>
          </div>
          <div className="ctaCard">
            <div>
              <strong>Download checklist</strong>
              <div className="muted">Prepare details before we call.</div>
            </div>
            <button className="btnSecondary">Download PDF</button>
          </div>
        </div>

        <div className="row">
          <button
            className="btn"
            onClick={() => {
              reset();
              navigate("/apply/step/1", { replace: true });
            }}
          >
            Start new submission
          </button>
        </div>
      </div>
    </div>
  );
}
