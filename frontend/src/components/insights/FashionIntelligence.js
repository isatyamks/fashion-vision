export default function FashionIntelligence({ attributes }) {
  if (!attributes) return null;

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-card)",
      padding: "var(--space-24)",
      marginBottom: "var(--space-32)"
    }}>
      <h3 style={{ fontSize: "18px", marginBottom: "var(--space-24)" }}>Semantic Feature Intelligence</h3>
      
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-16)" }}>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
          <span style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)" }}>Category</span>
          <span style={{ fontSize: "16px", fontWeight: "500", textTransform: "capitalize" }}>{attributes.category || "Unknown"}</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
          <span style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)" }}>Color Palette</span>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-8)" }}>
            <div style={{ 
              width: "16px", 
              height: "16px", 
              borderRadius: "50%", 
              background: attributes.color !== "unknown" ? attributes.color : "#ccc",
              border: "1px solid var(--border)"
            }}></div>
            <span style={{ fontSize: "16px", fontWeight: "500", textTransform: "capitalize" }}>{attributes.color || "Unknown"}</span>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
          <span style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)" }}>Detected Style</span>
          <span style={{ fontSize: "16px", fontWeight: "500", textTransform: "capitalize" }}>{attributes.style || "Unknown"}</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
          <span style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)" }}>Confidence</span>
          <span style={{ fontSize: "16px", fontWeight: "500", color: "#10b981" }}>
            {attributes.confidence ? `${(attributes.confidence * 100).toFixed(0)}%` : "N/A"}
          </span>
        </div>

      </div>
    </div>
  );
}
