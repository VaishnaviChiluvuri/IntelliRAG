import React from "react";

export default function AnswerCard({ result }) {
  if (!result) return null;

  const score = result.overall_prm_score ?? 0;
  let badgeClass = "score-badge score-weak";
  if (score >= 0.8) badgeClass = "score-badge score-strong";
  else if (score >= 0.6) badgeClass = "score-badge score-moderate";

  return (
    <section className="card answer-card">
      <div className="card-header">
        <h2>4. Final Verified Answer</h2>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {result.knowledge_source && (
            <span className="badge-neutral" style={{ fontWeight: 600 }}>
              {result.knowledge_source}
            </span>
          )}
          <div className={badgeClass}>
            {result.overall_support_level} · PRM: {score.toFixed(2)}
          </div>
        </div>
      </div>

      <p className="disclaimer-text">{result.score_disclaimer}</p>

      <div className="final-answer-box">{result.final_answer}</div>

      {result.wikipedia_fallback_used && (
        <div className="wiki-banner">
          <strong>External Knowledge Notice:</strong> {result.wikipedia_notice}
        </div>
      )}

      {result.document_citations?.length > 0 && (
        <div className="citations-block">
          <h3>Document Citations</h3>
          <div className="citation-chips">
            {result.document_citations.map((c, idx) => (
              <span key={idx} className="citation-chip">
                {c}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.external_citations?.length > 0 && (
        <div className="citations-block">
          <h3>External Citations (Wikipedia — Separate from Uploaded Document)</h3>
          {result.external_citations.map((ext, idx) => (
            <div key={idx} className="excerpt-item excerpt-external">
              <div className="excerpt-header">
                <span>{ext.source}: {ext.title}</span>
                <a href={ext.url} target="_blank" rel="noreferrer">
                  View Article ↗
                </a>
              </div>
              <p>{ext.summary}</p>
            </div>
          ))}
        </div>
      )}

      {result.supporting_excerpts?.length > 0 && (
        <div className="citations-block">
          <h3>Supporting Document Excerpts</h3>
          <div className="excerpts-list">
            {result.supporting_excerpts.map((ex, idx) => (
              <div key={idx} className="excerpt-item">
                <div className="excerpt-header">
                  <strong>{ex.citation}</strong>
                  <span>FAISS Similarity: {ex.similarity_score?.toFixed(3)}</span>
                </div>
                <p>{ex.excerpt}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}