import React from "react";
import ReactDOM from "react-dom/client";
import App from "./app/App";
import { FormProvider } from "./app/formStore";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <FormProvider>
      <App />
    </FormProvider>
  </React.StrictMode>
);
