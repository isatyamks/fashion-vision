"use client";

import { useState, useRef, useEffect } from "react";
import Navbar from "../components/layout/Navbar";
import UploadHero from "../components/search/UploadHero";
import VideoPlayer from "../components/video/VideoPlayer";
import ProductCard from "../components/product/ProductCard";
import FashionIntelligence from "../components/insights/FashionIntelligence";
import DynamicLoader from "../components/ui/DynamicLoader";

export default function Home() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [attributes, setAttributes] = useState(null);
  const [latency, setLatency] = useState(null);
  const [serverStatus, setServerStatus] = useState("loading");
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Poll the health endpoint until the server is ready
    const checkHealth = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/health");
        const data = await res.json();
        setServerStatus(data.status);
        if (data.status === "ready") {
          clearInterval(healthInterval);
        }
      } catch (err) {
        // server might be completely down
        setServerStatus("offline");
      }
    };

    checkHealth(); // Check immediately
    const healthInterval = setInterval(checkHealth, 2000); // Poll every 2 seconds

    return () => clearInterval(healthInterval);
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setResults([]);
      setAttributes(null);
      setLatency(null);
      
      // Auto-extract & match when file is selected
      handleMatch(file);
    }
  };

  const handleMatch = async (file) => {
    setLoading(true);
    setResults([]);
    setAttributes(null);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/api/match", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.matches) {
        setResults(data.matches);
      }
      if (data.extracted_attributes) {
        setAttributes(data.extracted_attributes);
      }
      if (data.latency) {
        setLatency(data.latency);
      }
    } catch (error) {
      console.error("Error running AI match:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-layout">
      <Navbar />

      <main className="app-container">
        {!videoUrl ? (
          <UploadHero 
            onUploadClick={() => fileInputRef.current?.click()} 
            serverStatus={serverStatus} 
          />
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "400px 1fr", gap: "var(--space-64)", height: "100%" }}>
            
            {/* Left Panel: Video */}
            <div style={{ 
              display: "flex", 
              flexDirection: "column", 
              height: "100%",
              overflowY: "auto",
              paddingBottom: "var(--space-64)",
              paddingRight: "var(--space-12)"
            }}>
              <VideoPlayer videoUrl={videoUrl} onFileSelect={handleFileSelect} />
              
              <div style={{ marginTop: "var(--space-32)", padding: "var(--space-16)", background: "var(--surface)", borderRadius: "var(--radius-card)", border: "1px solid var(--border)" }}>
                <h4 style={{ fontSize: "14px", fontWeight: "600", marginBottom: "8px" }}>Agentic Inference Pipeline</h4>
                <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>Autonomous neural agents processing 30fps video via YOLOv8 and Google SigLIP embeddings.</p>
                
              </div>


            </div>

            {/* Right Panel: Discovery */}
            <div style={{ 
              display: "flex", 
              flexDirection: "column", 
              height: "100%", 
              overflowY: "auto",
              paddingRight: "var(--space-16)",
              paddingBottom: "var(--space-64)"
            }}>
              
              {/* Loader */}
              {loading && <DynamicLoader />}

              {/* Results */}
              {!loading && results.length > 0 && (
                <div style={{ animation: "fadeIn 0.5s ease" }}>
                  <FashionIntelligence attributes={attributes} />
                  

                  <div style={{ 
                    display: "flex", 
                    justifyContent: "space-between", 
                    alignItems: "baseline",
                    marginBottom: "var(--space-24)",
                    borderBottom: "1px solid var(--border)",
                    paddingBottom: "var(--space-12)"
                  }}>
                    <h2 style={{ fontSize: "24px", fontFamily: "var(--font-serif, Georgia, serif)" }}>Resolved Inventory Matches</h2>
                    <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>{results.length} results</span>
                  </div>

                  <div style={{ 
                    display: "grid", 
                    gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", 
                    gap: "var(--space-32) var(--space-24)" 
                  }}>
                    {results.map((product, idx) => (
                      <ProductCard key={product.product_id} product={product} idx={idx} />
                    ))}
                  </div>

                  {/* Engine Telemetry Widget at Bottom */}
                  {latency && (
                    <div style={{ 
                      marginTop: "var(--space-64)", 
                      background: "transparent", 
                      borderRadius: "var(--radius-card)", 
                      padding: "var(--space-24)",
                      border: "1px solid var(--border)",
                      animation: "fadeIn 0.5s ease"
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
                        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#10B981" }}></div>
                        <h3 style={{ fontSize: "18px", fontWeight: "400", margin: 0, fontFamily: "var(--font-serif, Georgia, serif)", color: "var(--text)" }}>Processing Telemetry</h3>
                      </div>
                      
                      <div style={{ display: "flex", flexDirection: "column", gap: "14px", fontSize: "14px", color: "var(--text-secondary)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>YOLOv8 Subject Extraction</span>
                          <span style={{ color: "var(--text)" }}>{latency.yolo_cropping}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>Zero-Shot Semantic Color</span>
                          <span style={{ color: "var(--text)" }}>{latency.color_extraction}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>SigLIP Neural Encoding</span>
                          <span style={{ color: "var(--text)" }}>{latency.siglip_encoding}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>FAISS Vector Reranking</span>
                          <span style={{ color: "var(--text)" }}>{latency.faiss_search}</span>
                        </div>
                        
                        <div style={{ 
                          display: "flex", 
                          justifyContent: "space-between", 
                          alignItems: "center",
                          fontWeight: "500", 
                          marginTop: "8px", 
                          paddingTop: "16px", 
                          borderTop: "1px solid var(--border)" 
                        }}>
                          <span style={{ color: "var(--text)", fontSize: "16px" }}>Total Pipeline Latency</span>
                          <span style={{ color: "#10B981", fontSize: "16px" }}>{latency.total_time}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>
        )}

        {/* Hidden Input for Empty State */}
        <input 
          type="file" 
          accept="video/mp4,video/quicktime" 
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />
      </main>
    </div>
  );
}
