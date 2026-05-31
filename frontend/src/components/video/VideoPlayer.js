import { useRef } from "react";

export default function VideoPlayer({ videoUrl, onFileSelect }) {
  const fileInputRef = useRef(null);

  if (!videoUrl) return null;

  return (
    <div style={{
      width: "100%",
      maxHeight: "calc(100vh - 340px)",
      background: "transparent",
      position: "relative",
      display: "flex",
      justifyContent: "center",
      alignItems: "center"
    }}>
      <video 
        src={videoUrl} 
        controls 
        autoPlay 
        loop 
        muted
        style={{ 
          maxWidth: "100%", 
          maxHeight: "calc(100vh - 340px)", 
          objectFit: "contain",
          borderRadius: "var(--radius-card)",
          boxShadow: "var(--shadow-lg)"
        }}
      />
      
      {/* Change Video Button */}
      <button 
        onClick={() => fileInputRef.current.click()}
        style={{
          position: "absolute",
          top: "16px",
          right: "16px",
          background: "rgba(255,255,255,0.9)",
          border: "none",
          padding: "8px 16px",
          borderRadius: "20px",
          cursor: "pointer",
          fontSize: "12px",
          fontWeight: "600",
          boxShadow: "var(--shadow-sm)"
        }}
      >
        Change Video
      </button>
      <input 
        type="file" 
        accept="video/mp4,video/quicktime" 
        ref={fileInputRef}
        onChange={onFileSelect}
        style={{ display: "none" }}
      />
    </div>
  );
}
