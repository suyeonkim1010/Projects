export default function Field({ label, name, register, error, type = "text", placeholder }) {
  const errorId = error ? `${name}-error` : undefined;

  return (
    <div className="field">
      <label className="label" htmlFor={name}>{label}</label>
      <input
        id={name}
        name={name}
        className={`input ${error ? "inputError" : ""}`}
        {...register(name)}
        type={type}
        placeholder={placeholder}
        autoComplete="on"
        aria-invalid={Boolean(error)}
        aria-describedby={errorId}
      />
      {error ? (
        <div className="fieldError" id={errorId} role="alert">
          {error}
        </div>
      ) : null}
    </div>
  );
}
