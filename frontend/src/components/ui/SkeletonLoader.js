export default function SkeletonLoader() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "var(--space-32) var(--space-24)" }}>
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div key={i} style={{ display: "flex", flexDirection: "column", gap: "var(--space-12)" }}>
          <div className="skeleton" style={{ width: "100%", aspectRatio: "3/4", borderRadius: "var(--radius-img)" }}></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-8)" }}>
            <div className="skeleton" style={{ width: "40%", height: "14px", borderRadius: "4px" }}></div>
            <div className="skeleton" style={{ width: "90%", height: "14px", borderRadius: "4px" }}></div>
            <div className="skeleton" style={{ width: "60%", height: "14px", borderRadius: "4px" }}></div>
          </div>
        </div>
      ))}
    </div>
  );
}
