import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { FormProvider } from "./store/formStore.jsx";
import "./styles/wizard.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <FormProvider>
      <App />
    </FormProvider>
  </React.StrictMode>
);
