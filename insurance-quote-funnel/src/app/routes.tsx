import { Routes, Route, Navigate } from "react-router-dom";
import QuotePage from "../pages/Quote/QuotePage";
import ThankYouPage from "../pages/ThankYou/ThankYouPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/apply/step/1" replace />} />
      <Route path="/apply/step/:step" element={<QuotePage />} />
      <Route path="/apply/thank-you" element={<ThankYouPage />} />
      <Route path="*" element={<div>404</div>} />
    </Routes>
  );
}
