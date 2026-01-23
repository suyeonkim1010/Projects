import { useState } from "react";
import { useLocation, Navigate, useNavigate } from "react-router-dom";
import { useFormStore } from "../store/formStore.jsx";

export default function ThankYou() {
  const location = useLocation();
  const navigate = useNavigate();
  const { data, isSubmissionLocked, reset } = useFormStore();
  const [showModal, setShowModal] = useState(false);

  if (!data.submitted) {
    return <Navigate to="/apply/step/1" replace />;
  }

  const confirmationId = location.state?.confirmationId || data.confirmationId;

  const handleDownload = () => {
    const pdfContent =
      "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 120>>stream\nBT /F1 18 Tf 72 720 Td (Insurance Checklist) Tj 0 -28 Td /F1 12 Tf (1. Gather policy history) Tj 0 -18 Td (2. List recent claims) Tj 0 -18 Td (3. Confirm driver details) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000117 00000 n\n0000000244 00000 n\n0000000421 00000 n\ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n505\n%%EOF";
    const blob = new Blob([pdfContent], { type: "application/pdf" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "insurance-checklist.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

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
            <button className="btn" type="button" onClick={() => setShowModal(true)}>
              Schedule request
            </button>
          </div>
          <div className="ctaCard">
            <div>
              <strong>Download checklist</strong>
              <div className="muted">Prepare details before we call.</div>
            </div>
            <button className="btnSecondary" type="button" onClick={handleDownload}>
              Download PDF
            </button>
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

      {showModal ? (
        <div className="modal" role="dialog" aria-modal="true">
          <div className="modalCard">
            <h2>Thank you for your request!!</h2>
            <p>We will email you with available time slots shortly.</p>
            <button className="btn" type="button" onClick={() => setShowModal(false)}>
              Close
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
