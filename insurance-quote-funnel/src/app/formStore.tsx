import React, { createContext, useContext, useEffect, useState } from "react";
import type { QuoteFormData } from "../pages/Quote/quoteSchema";

type StoredFormData = {
  values: Partial<QuoteFormData>;
  submitted: boolean;
  submittedAt?: number;
  confirmationId?: string;
};

type FormContextValue = {
  data: StoredFormData;
  setData: React.Dispatch<React.SetStateAction<StoredFormData>>;
};

const FormContext = createContext<FormContextValue | null>(null);
const STORAGE_KEY = "apply_form_v1";

const defaultData: StoredFormData = {
  values: {},
  submitted: false
};

export function FormProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<StoredFormData>(() => {
    if (typeof window === "undefined") return defaultData;
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return defaultData;
    try {
      return JSON.parse(saved) as StoredFormData;
    } catch {
      return defaultData;
    }
  });

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }, [data]);

  return <FormContext.Provider value={{ data, setData }}>{children}</FormContext.Provider>;
}

export function useFormStore() {
  const ctx = useContext(FormContext);
  if (!ctx) throw new Error("Wrap with <FormProvider>");
  return ctx;
}
