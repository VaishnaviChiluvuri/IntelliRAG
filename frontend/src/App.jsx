import React, { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import PipelineTracker from "./components/PipelineTracker.jsx";
import AnswerCard from "./components/AnswerCard.jsx";
import ReasoningDetails from "./components/ReasoningDetails.jsx";

const SAMPLE_QUESTIONS = [
  "What is the main topic, and what are the key findings or qualifications described?",
  "What skills or concepts are introduced, and how are they applied in practice?",
  "Does the subject meet the specified requirements based on the document criteria?"
];

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [docInfo, setDocInfo] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null);
  const [pipelineState, setPipelineState] = useState({
    stage: "idle",
    detail: "Upload a document and submit a question to begin.",
    completed_stages: []
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    checkHealth();
    const timer = setInterval(checkHealth, 8000);
    return () => clearInterval(timer);
  }, []);

  const handleUploadSuccess = (data) => {
    setDocInfo(data);
    setResult(null);
    setErrorMsg("");
    setPipelineState({
      stage: "idle",
      detail: `Ready to query "${data.filename}".`,
      completed_stages: []
    });
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!docInfo?.session_id) {
      setErrorMsg("Please upload and index a document first.");
      return;
    }
    const trimmed = question.trim();
    if (!trimmed) {
      setErrorMsg("Please enter a question.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setResult(null);

    const requestId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `req_${Date.now()}`;

    setPipelineState({
      stage: "document_retrieval",
      detail: "Submitting question to multi-hop reasoning engine...",
      completed_stages: []
    });

    const pollTimer = setInterval(async () => {
      try {
        const sRes = await fetch(`${API_BASE}/status/${requestId}`);
        if (sRes.ok) {
          const sData = await sRes.json();
          if (sData.stage && sData.stage !== "idle") {
            setPipelineState(sData);
          }
        }
      } catch {
        // Ignore transient polling errors while /ask runs
      }
    }, 450);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: docInfo.session_id,
          question: trimmed,
          request_id: requestId
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to generate answer.");
      }
      setResult(data);
      setPipelineState({
        stage: "completed",
        detail: "All multi-hop reasoning and PRM/PEM stages completed.",
        completed_stages: [
          "document_retrieval",
          "question_decomposition",
          "evidence_extraction",
          "intermediate_reasoning",
          "prm_evaluation",
          "pem_correction",
          "final_answer_generation"
        ]
      });
    } catch (err) {
      setErrorMsg(err.message || "Backend request failed.");
      setPipelineState((prev) => ({
        ...prev,
        stage: "error",
        detail: err.message || "Error during execution"
      }));
    } finally {
      clearInterval(pollTimer);
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Document Multi-Hop RAG</h1>
          <p className="subtitle">
            Evidence-based multi-step question answering with ReARTeR-inspired PRM &amp; PEM
          </p>
        </div>
        <div className={`conn-pill ${backendOnline ? "conn-online" : "conn-offline"}`}>
          <span className="conn-dot" />
          {backendOnline ? "Backend Connected (localhost:8000)" : "Backend Offline"}
        </div>
      </header>

      <main className="main-container">
        <div className="top-grid">
          <UploadPanel onUploadSuccess={handleUploadSuccess} docInfo={docInfo} />

          <section className="card">
            <div className="card-header">
              <h2>2. Multi-Hop Question</h2>
              <span className="badge-neutral">
                {docInfo ? `Active: ${docInfo.filename}` : "No document loaded"}
              </span>
            </div>

            <form onSubmit={handleAsk} className="question-form">
              <textarea
                rows={3}
                className="question-input"
                placeholder={
                  docInfo
                    ? "Ask a multi-step analytical or policy question about the uploaded document..."
                    : "Upload a PDF, DOCX, or TXT document first to enable questions..."
                }
                disabled={!docInfo || loading}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />

              <div className="sample-prompts">
                <span className="sample-label">Sample test queries:</span>
                {SAMPLE_QUESTIONS.map((sq, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="sample-chip"
                    disabled={!docInfo || loading}
                    onClick={() => setQuestion(sq)}
                  >
                    Q{idx + 1}: {sq}
                  </button>
                ))}
              </div>

              <div className="form-actions">
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={!docInfo || loading || !question.trim()}
                >
                  {loading ? "Running Multi-Hop Reasoning..." : "Ask Document"}
                </button>
              </div>
            </form>

            {errorMsg && <div className="status-error">✕ {errorMsg}</div>}
          </section>
        </div>

        <PipelineTracker pipelineState={pipelineState} />

        {result && (
          <>
            <AnswerCard result={result} />
            <ReasoningDetails steps={result.reasoning_steps} />
          </>
        )}
      </main>
    </div>
  );
}