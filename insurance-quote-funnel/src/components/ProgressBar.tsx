type Props = {
  current: number; // 1-based
  total: number;
  labels?: string[];
};

export default function ProgressBar({ current, total, labels }: Props) {
  const pct = Math.round((current / total) * 100);

  return (
    <div style={{ margin: "16px 0 24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ fontWeight: 700 }}>
          Step {current} of {total}
        </div>
        <div style={{ color: "#555" }}>{pct}%</div>
      </div>

      <div style={{ height: 10, background: "#eee", borderRadius: 999 }}>
        <div
          style={{
            height: 10,
            width: `${pct}%`,
            background: "#111",
            borderRadius: 999,
            transition: "width 200ms ease"
          }}
        />
      </div>

      {labels && (
        <div
          style={{
            display: "flex",
            gap: 12,
            marginTop: 10,
            flexWrap: "wrap",
            color: "#555"
          }}
        >
          {labels.map((l, i) => (
            <span key={l} style={{ fontWeight: i + 1 === current ? 700 : 400 }}>
              {i + 1}. {l}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
