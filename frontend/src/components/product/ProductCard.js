import Image from "next/image";

export default function ProductCard({ product, idx }) {
  // Normalize match percentage for UI (Bounded between 75% and 99%)
  // The raw hybrid score is usually between 0.2 and 0.8
  const rawScore = product.confidence;
  const matchPercentage = Math.min(99, Math.max(75, 75 + (rawScore * 24))).toFixed(0);

  let matchTag = "Similar Style";
  if (matchPercentage > 95) matchTag = "Best Match";
  else if (matchPercentage > 85) matchTag = "Great Match";

  return (
    <div 
      style={{
        display: "flex",
        flexDirection: "column",
        cursor: "pointer",
        animation: `fadeIn 0.5s ease forwards`,
        animationDelay: `${idx * 0.1}s`,
        opacity: 0,
        position: "relative"
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "scale(1.02)";
        e.currentTarget.style.boxShadow = "var(--shadow-md)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Internal CSS for animation */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Image Container */}
      <div style={{
        width: "100%",
        aspectRatio: "3/4",
        position: "relative",
        background: "#F0F0F0",
        borderRadius: "var(--radius-img)",
        overflow: "hidden",
        marginBottom: "var(--space-12)"
      }}>
        <Image 
          src={product.image_url} 
          alt={product.title}
          fill
          style={{ objectFit: "cover" }}
          unoptimized
        />
        {/* Wishlist Button Overlay */}
        <div style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          background: "rgba(255,255,255,0.9)",
          borderRadius: "50%",
          width: "32px",
          height: "32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "var(--shadow-sm)"
        }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
          </svg>
        </div>
      </div>

      {/* Product Info */}
      <div style={{ padding: "0 var(--space-4)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-4)" }}>
          <span style={{ fontSize: "12px", fontWeight: "600", textTransform: "uppercase", color: "var(--text-secondary)" }}>
            Virgio
          </span>
          <span style={{ fontSize: "12px", color: "var(--accent)", fontWeight: "600" }}>
            {matchPercentage}% • {matchTag}
          </span>
        </div>
        
        <h4 style={{ fontSize: "14px", fontWeight: "400", lineHeight: "1.4", color: "var(--text-primary)", marginBottom: "var(--space-8)" }}>
          {product.title}
        </h4>
        
        <div>
          <span style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>₹{product.price}</span>
          {product.mrp && product.mrp !== "0" && product.mrp !== product.price && (
            <span style={{ fontSize: "12px", color: "var(--text-secondary)", textDecoration: "line-through", marginLeft: "var(--space-8)" }}>
              ₹{product.mrp}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
