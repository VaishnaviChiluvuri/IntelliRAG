import React from "react";

const STAGES = [
  { key: "document_retrieval", label: "1. Document retrieval" },
  { key: "question_decomposition", label: "2. Question decomposition" },
  { key: "evidence_extraction", label: "3. Evidence extraction" },
  { key: "intermediate_reasoning", label: "4. Intermediate reasoning" },
  { key: "prm_evaluation", label: "5. PRM evaluation" },
  { key: "pem_correction", label: "6. PEM correction (if required)" },
  { key: "final_answer_generation", label: "7. Final answer generation" }
];

export default function PipelineTracker({ pipelineState }) {
  const { stage, detail, completed_stages = [] } = pipelineState;
  const isAllDone = stage === "completed";

  return (
    <section className="card">
      <div className="card-header">
        <h2>3. Reasoning Pipeline Status</h2>
        <span className="status-caption">{detail || "Idle — awaiting question"}</span>
      </div>

      <div className="pipeline-grid">
        {STAGES.map((st) => {
          const isCompleted = isAllDone || completed_stages.includes(st.key);
          const isActive = !isAllDone && stage === st.key;
          let cls = "stage-pill stage-pending";
          if (isCompleted) cls = "stage-pill stage-done";
          else if (isActive) cls = "stage-pill stage-active";

          return (
            <div key={st.key} className={cls}>
              <span className="stage-dot">
                {isCompleted ? "✓" : isActive ? "●" : "○"}
              </span>
              <span>{st.label}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}