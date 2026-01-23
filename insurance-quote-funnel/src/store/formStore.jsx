import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const FormContext = createContext(null);

const STORAGE_KEY = "apply_form_v1";
const SUBMIT_LOCK_TTL_MS = 24 * 60 * 60 * 1000;

function safeParse(json) {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

const initialData = {
  step1: { fullName: "" },
  step2: { email: "", phone: "" },
  step3: { experience: "" },
  step4: { notes: "" },
  submitted: false,
  submittedAt: null,
  confirmationId: null,
  lastSavedAt: null,
  lastError: null
};

export function FormProvider({ children }) {
  const [data, setData] = useState(() => {
    const saved = safeParse(localStorage.getItem(STORAGE_KEY));
    return saved ? { ...initialData, ...saved } : initialData;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }, [data]);

  const setDataWithTimestamp = useCallback((updater) => {
    setData((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      return { ...next, lastSavedAt: Date.now() };
    });
  }, []);

  const isSubmissionLocked = useMemo(() => {
    if (!data.submitted || !data.submittedAt) return false;
    return Date.now() - data.submittedAt < SUBMIT_LOCK_TTL_MS;
  }, [data.submitted, data.submittedAt]);

  const value = useMemo(
    () => ({
      data,
      setData: setDataWithTimestamp,
      isSubmissionLocked,
      reset: () => {
        localStorage.removeItem(STORAGE_KEY);
        setDataWithTimestamp(initialData);
      }
    }),
    [data, isSubmissionLocked]
  );

  return <FormContext.Provider value={value}>{children}</FormContext.Provider>;
}

export function useFormStore() {
  const ctx = useContext(FormContext);
  if (!ctx) throw new Error("Wrap app with <FormProvider>.");
  return ctx;
}
