import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import ApplyWizard from "./routes/ApplyWizard.jsx";
import ThankYou from "./routes/ThankYou.jsx";
import { Toaster } from "react-hot-toast";

export default function App() {
  return (
    <BrowserRouter>
      <Toaster position="top-right" />
      <Routes>
        <Route path="/" element={<Navigate to="/apply/step/1" replace />} />
        <Route path="/apply/step/:step" element={<ApplyWizard />} />
        <Route path="/apply/thank-you" element={<ThankYou />} />
        <Route path="*" element={<Navigate to="/apply/step/1" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
