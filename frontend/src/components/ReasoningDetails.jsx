import React, { useState } from "react";

export default function ReasoningDetails({ steps }) {
  const [expanded, setExpanded] = useState(true);

  if (!steps || steps.length === 0) return null;

  const getScoreClass = (score) => {
    if (score >= 0.8) return "score-badge score-strong";
    if (score >= 0.6) return "score-badge score-moderate";
    return "score-badge score-weak";
  };

  return (
    <section className="card">
      <div className="card-header">
        <h2>5. Multi-Hop Reasoning, PRM Evaluation &amp; PEM Correction Trace</h2>
        <button
          type="button"
          className="btn-secondary"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Collapse Steps" : `Expand Steps (${steps.length})`}
        </button>
      </div>

      {expanded && (
        <div className="steps-container">
          {steps.map((step) => (
            <div key={step.step_number} className="step-card">
              <div className="step-top">
                <div className="step-title">
                  <span className="step-num">Hop {step.step_number}</span>
                  <strong>{step.sub_question}</strong>
                </div>
                <div className="step-badges">
                  <span className={getScoreClass(step.prm_score)}>
                    Initial PRM: {step.prm_score.toFixed(2)} ({step.support_level})
                  </span>
                  <span
                    className={`status-tag ${
                      step.status === "Unresolved" ? "tag-unresolved" : "tag-accepted"
                    }`}
                  >
                    {step.status}
                  </span>
                </div>
              </div>

              <div className="step-body">
                <div className="step-row">
                  <span className="field-label">Intermediate Answer:</span>
                  <span>{step.intermediate_answer}</span>
                </div>

                <div className="step-row">
                  <span className="field-label">PRM Evaluation:</span>
                  <span>{step.evaluation_explanation}</span>
                </div>

                {step.detected_issue && (
                  <div className="step-row issue-text">
                    <span className="field-label">Detected Issue:</span>
                    <span>{step.detected_issue}</span>
                  </div>
                )}

                {step.pem_correction && step.pem_correction.attempted && (
                  <div className="pem-box">
                    <div className="pem-header">
                      PEM Correction Triggered — {step.pem_correction.status}
                    </div>
                    <p>
                      <strong>Diagnosis:</strong> {step.pem_correction.error_explanation}
                    </p>
                    <p>
                      <strong>Corrected Answer:</strong> {step.pem_correction.corrected_answer}
                    </p>
                    <p>
                      <strong>Re-Evaluation PRM Score:</strong>{" "}
                      {step.pem_correction.re_eval_score.toFixed(2)} —{" "}
                      {step.pem_correction.re_eval_explanation}
                    </p>
                  </div>
                )}

                <div className="step-row final-step-row">
                  <span className="field-label">Final Step Conclusion:</span>
                  <strong>{step.final_answer}</strong>
                </div>

                {step.retrieved_evidence?.length > 0 && (
                  <details className="evidence-details">
                    <summary>
                      Retrieved Evidence ({step.retrieved_evidence.length} chunk
                      {step.retrieved_evidence.length > 1 ? "s" : ""})
                    </summary>
                    <ul>
                      {step.retrieved_evidence.map((ev, i) => (
                        <li key={i}>
                          <strong>{ev.citation}</strong> (sim: {ev.similarity_score?.toFixed(3)}):{" "}
                          {ev.text}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}