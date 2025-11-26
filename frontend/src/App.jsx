import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [condition, setCondition] = useState(3); // 3 = UI/app designs
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f || null);
    setResultUrl(null);
    setError("");
    if (f) setPreviewUrl(URL.createObjectURL(f));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Select a screenshot first.");
      return;
    }
    setLoading(true);
    setError("");
    setResultUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("condition", condition);

      const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || "Request failed");
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "24px" }}>
      <h1>UI Saliency Heatmap (SUM)</h1>

      <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
        <div style={{ marginBottom: 12 }}>
          <label>
            Upload UI screenshot (PNG/JPG)
            <br />
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Condition:
            <select
              value={condition}
              onChange={(e) => setCondition(Number(e.target.value))}
              style={{ marginLeft: 8 }}
            >
              <option value={0}>0 – Natural images</option>
              <option value={1}>1 – Text / documents</option>
              <option value={2}>2 – Web pages</option>
              <option value={3}>3 – UI / app designs</option>
            </select>
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Generating..." : "Generate Heatmap"}
        </button>
      </form>

      {error && (
        <div style={{ color: "red", marginBottom: 16 }}>
          {error}
        </div>
      )}

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start" }}>
        {previewUrl && (
          <div>
            <h3>Input</h3>
            <img
              src={previewUrl}
              alt="input"
              style={{ maxWidth: 300, borderRadius: 8 }}
            />
          </div>
        )}

        {resultUrl && (
          <div>
            <h3>Overlay Result</h3>
            <img
              src={resultUrl}
              alt="heatmap overlay"
              style={{ maxWidth: 300, borderRadius: 8 }}
            />
            <div style={{ marginTop: 8 }}>
              <a href={resultUrl} download="heatmap_overlay.png">
                Download PNG
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
