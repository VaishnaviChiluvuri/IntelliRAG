import React, { useRef, useState } from "react";

const ALLOWED_EXTS = [".pdf", ".docx", ".txt"];
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function UploadPanel({ onUploadSuccess, docInfo }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const formatBytes = (bytes) => {
    if (!bytes && bytes !== 0) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleFile = async (file) => {
    if (!file) return;
    setErrorMsg("");
    setStatusMsg("");

    const lower = file.name.toLowerCase();
    const valid = ALLOWED_EXTS.some((ext) => lower.endsWith(ext));
    if (!valid) {
      setErrorMsg("Unsupported file format. Please select a PDF, DOCX, or TXT file.");
      return;
    }
    if (file.size === 0) {
      setErrorMsg("Selected file is empty (0 bytes).");
      return;
    }

    setSelectedFile(file);
    setUploading(true);
    setStatusMsg("Uploading, extracting text, generating embeddings, and building FAISS index...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed.");
      }

      setStatusMsg(
        `Indexed ${data.chunk_count} chunks${
          data.page_count ? ` across ${data.page_count} page(s)` : ""
        }. Ready for multi-hop questions.`
      );
      onUploadSuccess(data);
    } catch (err) {
      setErrorMsg(err.message || "Failed to connect to backend during upload.");
      setStatusMsg("");
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <section className="card">
      <div className="card-header">
        <h2>1. Document Upload &amp; Indexing</h2>
        <span className="badge-neutral">PDF · DOCX · TXT</span>
      </div>

      <div
        className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="dropzone-title">
          Drag &amp; drop your document here, or <span className="link-accent">Browse Files</span>
        </p>
        <p className="dropzone-sub">
          Supported formats: PDF (with page citations), DOCX, TXT · Max size: 25 MB
        </p>
      </div>

      {(selectedFile || docInfo) && (
        <div className="file-meta-bar">
          <div>
            <strong>File:</strong> {docInfo?.filename || selectedFile?.name}
          </div>
          <div>
            <strong>Size:</strong>{" "}
            {formatBytes(docInfo?.file_size_bytes ?? selectedFile?.size)}
          </div>
          {docInfo?.session_id && (
            <div className="mono-small">
              <strong>Session:</strong> {docInfo.session_id.slice(0, 8)}...
            </div>
          )}
        </div>
      )}

      {uploading && <div className="status-info">⏳ {statusMsg}</div>}
      {!uploading && statusMsg && <div className="status-success">✓ {statusMsg}</div>}
      {errorMsg && <div className="status-error">✕ {errorMsg}</div>}
    </section>
  );
}