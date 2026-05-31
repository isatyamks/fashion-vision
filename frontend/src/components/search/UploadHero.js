export default function UploadHero({ onUploadClick, serverStatus }) {
  const isReady = serverStatus === "ready";
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      textAlign: "center",
      gap: "var(--space-32)"
    }}>
      <h1 style={{ maxWidth: "700px", margin: "0 auto", fontSize: "42px", fontWeight: "700", letterSpacing: "-1px" }}>Enterprise-Grade Agentic Visual Search Engine</h1>
      
      <p style={{ maxWidth: "600px", margin: "0 auto", fontSize: "18px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
        Autonomous YOLOv8 vision agents extract subjects in real-time. Zero-shot semantic pipelines and FAISS vector reranking guarantee exact inventory matching.
      </p>

      <button 
        className="btn-primary" 
        style={{ 
          padding: "var(--space-16) var(--space-48)", 
          fontSize: "18px",
          opacity: isReady ? 1 : 0.6,
          cursor: isReady ? "pointer" : "not-allowed"
        }}
        onClick={isReady ? onUploadClick : undefined}
        disabled={!isReady}
      >
        {isReady ? "Initialize Inference Pipeline" : "Waking up AI Vision Models..."}
      </button>

      {/* Modern Horizontal Timeline */}
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        gap: "var(--space-16)", 
        marginTop: "var(--space-64)",
        color: "var(--text-secondary)",
        fontSize: "14px",
        fontWeight: "500"
      }}>
        <span>Video Ingestion</span>
        <span>→</span>
        <span>Neural Extraction</span>
        <span>→</span>
        <span>Vector Reranking</span>
        <span>→</span>
        <span>Inventory Resolution</span>
      </div>
    </div>
  );
}
