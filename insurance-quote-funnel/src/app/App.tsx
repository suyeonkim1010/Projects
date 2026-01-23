import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ maxWidth: 900, margin: "0 auto", padding: 20 }}>
        <AppRoutes />
      </div>
    </BrowserRouter>
  );
}
