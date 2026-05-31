import { useState, useEffect } from "react";

const logs = [
  "Initializing AI Vision Agent...",
  "Extracting frames via YOLOv8...",
  "Deduplicating identical crops...",
  "Computing SigLIP visual embeddings...",
  "Running Zero-Shot NLP extraction...",
  "Performing FAISS nearest-neighbor search...",
  "Downloading candidate product images...",
  "Running K-Means Pixel Color Analyzer...",
  "Applying mathematical RGB distance penalty...",
  "Finalizing Hybrid Ranking..."
];

export default function DynamicLoader() {
  const [logIndex, setLogIndex] = useState(0);

  useEffect(() => {
    // Cycle through logs every 800ms to simulate the backend pipeline
    const interval = setInterval(() => {
      setLogIndex((prev) => (prev < logs.length - 1 ? prev + 1 : prev));
    }, 800);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      padding: "var(--space-64) 0",
      color: "var(--text-secondary)"
    }}>
      <div className="spinner" style={{ 
        width: "40px", 
        height: "40px", 
        border: "3px solid var(--border)", 
        borderTopColor: "var(--text-primary)", 
        borderRadius: "50%",
        animation: "spin 1s linear infinite",
        marginBottom: "var(--space-24)"
      }}></div>
      
      <div style={{
        background: "var(--surface)",
        padding: "var(--space-12) var(--space-24)",
        borderRadius: "var(--radius-btn)",
        border: "1px solid var(--border)",
        fontFamily: "monospace",
        fontSize: "12px",
        boxShadow: "var(--shadow-sm)",
        transition: "all 0.3s ease"
      }}>
        {">"} {logs[logIndex]}
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
