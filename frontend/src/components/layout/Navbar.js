"use client";
import Link from "next/link";

export default function Navbar() {
  return (
    <nav style={{
      position: "sticky",
      top: 0,
      zIndex: 100,
      background: "var(--surface)",
      borderBottom: "1px solid var(--border)",
      padding: "var(--space-16) var(--space-48)",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center"
    }}>
      {/* Left: Brand */}
      <Link href="/" style={{ textDecoration: "none" }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <h2 style={{ fontSize: "24px", margin: 0, color: "var(--text-primary)" }}>Fashion Vision</h2>
          <span style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)" }}>
            Visual Search for Fashion
          </span>
        </div>
      </Link>

      {/* Center: Navigation */}
      <div style={{ display: "flex", gap: "var(--space-32)", alignItems: "center" }}>
        <Link href="/" style={{ textDecoration: "none", color: "var(--text-primary)", fontWeight: "500", fontSize: "14px" }}>Product Matcher</Link>
      </div>

      {/* Right: Empty for balance */}
      <div style={{ width: "100px" }}></div>
    </nav>
  );
}
