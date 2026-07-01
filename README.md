# EarningsSignal Agent — LLM-Driven Alpha Factor Discovery Pipeline

An autonomous multi-agent system for discovering and iteratively refining predictive signals from earnings call transcripts, targeting AI Agent engineer roles at quantitative hedge funds.

## Overview

What distinguishes this project from conventional LLM factor mining:

| Standard Approach | This Project |
|---|---|
| LLM freely generates factors | **RAG + 6 theoretical clusters constrain the hypothesis space** |
| No quality governance | **Explicit multi-metric governance: IC / t-stat / zero_ratio / direction_consistency** |
| Trial-and-error iteration | **DiagnosisAgent performs root-cause analysis and generates targeted repair specs** |
| Black-box factor production | **Every decision trace recorded in feature_history.jsonl** |

## Architecture

```
Phase 0 (One-time)               Phase 1 (Agent Loop)
─────────────────────────────────────────────────────────
                                 ┌──────────────────┐
   Papers (20 PDFs) ──▶ RAG index│  HypothesisAgent  │←── repair_spec
                                 │  theory→hypothesis│    (from Diagnosis)
                                 └────────┬─────────┘
                                          │ feature_spec (JSON)
                                 ┌────────▼─────────┐
   Transcript batches ──▶ chunk │  ExtractionAgent  │
   retrieval (GPU matmul)       │  LLM batch scoring │
                                 └────────┬─────────┘
                                          │ feature_df (CSV)
                                 ┌────────▼─────────┐
   sp500_events.parquet ──▶ merge│ ValidationAgent   │
                                 │  IC · t-stat · NW │
                                 └────────┬─────────┘
                                          │ val_result (dict)
                                 ┌────────▼─────────┐
                                 │ GovernanceAgent    │
                                 │  G1–G4 hard rules  │
                                 └───┬──────────┬────┘
                               PASS  │          │ FAIL
                                     │   ┌──────▼──────────┐
                                     │   │  DiagnosisAgent   │
                                     │   │  RAG query → LLM  │
                                     │   │  root_cause + fix │
                                     │   └──────┬───────────┘
                                     │          │ _repair_spec
                                     │   ┌──────▼───────────┐
                                     │   │  Back to          │
                                     │   │  HypothesisAgent  │
                                     │   └──────────────────┘
                              ┌──────▼──────┐
                              │  fusion_eval │
                              │  multi-factor│
                              └──────────────┘
```

## Key Innovations

### 1. RAG-Constrained Hypothesis Generation
HypothesisAgent retrieves relevant theory from 20 academic papers (indexed via BGE-M3 embeddings) across 6 domain-specific clusters: tone_sentiment, information_asymmetry, forward_guidance, qa_subjectivity, managerial_behavior, alpha_discovery. The LLM reads the theory FIRST, then derives a feature specification — not the other way around.

### 2. Explicit Multi-Metric Governance
GovernanceAgent enforces four hard rules, each with a clear PASS/FAIL rationale:
- **G1**: IC magnitude (>|0.015|)
- **G2**: zero_ratio (uniform ≤0.70, concentrated ≤0.45) — with automated zero-type classification
- **G3**: |t-stat| ≥ 1.5
- **G4**: direction_consistency ≥ 60%

Every rule is human-auditable. No black-box quality scores.

### 3. Autonomous Diagnosis & Repair Loop
When a feature fails governance, DiagnosisAgent:
1. Maps failure symptoms to retrieval queries (no free-form LLM hallucination)
2. Searches the theory index for relevant evidence
3. Produces structured diagnosis: `{root_cause, fix, avoid, rag_refs}`
4. If the feature meets salvage criteria (IC>0.08, |t|>2.0, only G2 failure), generates a repair spec that feeds back to HypothesisAgent

**Real example**: `guidance_revision_direction` v1 → DiagnosisAgent identified "extraction threshold too strict for small-magnitude adjustments" → v2 expanded examples → v3 relaxed condition_scope. Three generations with extraction instruction growing from 159 to 636 characters, each failure driving a targeted fix.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
echo "SILICONFLOW_API_KEY=your_key_here" > .env

# 3. Run Phase 0 (one-time, builds FAISS indices from transcripts and papers)
#    See phase0_pipeline/README.md for detailed instructions
cd phase0_pipeline
pip install -r requirements.txt
python build_transcript_index.py      # transcript chunk index (~95万 embeddings, ~3.9GB)
python build_theory_index.py          # theory paper index (20 papers)

# 4. Run the Agent loop
cd ..
python run_agent.py --max-iter 20
# Output: agent_output/{feature_name}.csv, feature_history.jsonl

# 5. Evaluate fusion performance
python fusion_eval.py
# Output: agent_output/fusion_eval_results.csv
```

## Evaluation Results (S&P 500, 2021–2023 test period, industry-neutralized)

### Yearly IC Breakdown — Text Features Complement Financials

M0 (BASE13) is already a strong baseline built from 13 financial + momentum factors. Text features don't uniformly boost IC — they provide **orthogonal information when financial ratios are least reliable**:

| Period | M0 (BASE13) | + Text (7 selected) | Δ | Regime |
| ------ | ----------- | ------------------- | --- | ------ |
| **2020 (val)** | +0.0673 | **+0.0828** | **+0.0155 (+23%)** | COVID: financials distorted, text captures uncertainty |
| 2021 (test) | **+0.0376** | +0.0313 | −0.0063 | Recovery: strong earnings rebound favors financials |
| 2022 (test) | +0.1911 | **+0.2009** | +0.0098 | Bear market: text captures management tone shift |
| 2023 (test) | +0.0840 | **+0.1007** | +0.0167 | AI rally: text detects narrative divergence |

### Residual IC Proof — Information Orthogonal to Financials

To verify text features are not repackaging financial information, we compute the **residual IC**: each text feature's correlation with `move_post − M0_pred` (the part of future returns unexplained by BASE13).

Forward-looking text features show strong residual signal:

- `forward_numeric_specificity`: residual IC = **+0.108** (3× the average BASE13 factor marginal IC)
- `qa_spontaneity`: residual IC = **+0.043** (captures Q&A session dynamics absent from financial statements)
- `guidance_revision_direction` (v3): residual IC = **+0.033** (iterative repair improved from v1 baseline of +0.007)

These residual ICs confirm that earnings call transcripts contain **alpha information structurally absent from financial statements** — management tone, forward guidance framing, and Q&A spontaneity that accounting ratios cannot capture.

### Risk Diversification

Even when average IC is comparable, text features improve portfolio stability:

| Metric | M0 (BASE13) | + Text (13 features) | Improvement |
| ------ | ----------- | -------------------- | ----------- |
| Long-Short IR | 2.484 | **2.733** | **+10.0%** |
| Max Drawdown | −1.37% | **−0.49%** | **−64%** |
| NW t-stat | 5.075 | **5.674** | **+11.8%** |

Text features act as a **diversifying information source** — their prediction errors are less correlated with financial-factor errors, reducing peak-to-trough drawdowns. The same IC with lower volatility and lower drawdown is a genuine improvement in risk-adjusted performance.

## Limitations & Future Work

- **Single transcript source**: Framework currently tested on earnings calls only; migration to analyst reports, 10-K filings, and FOMC minutes is architecturally supported but not yet executed
- **Coverage dependency**: ~69% of S&P 500 episodes have transcript data; uncovered samples rely on BASE13 financial features
- **LLM API cost**: ~10M tokens per extraction run (DeepSeek V4 Flash, ~$0.14/1M input tokens)
- **Pool size**: 337 stocks (constrained by S&P 500 index membership and ex991 transcript availability)

## Project Structure

```
Fullproject/
├── agent_core/                    # Five-layer agent pipeline
│   ├── config.py                  # Shared paths, API keys, constants
│   ├── hypothesis_agent.py        # RAG-driven feature hypothesis generation
│   ├── extraction_agent.py        # LLM batch scoring (DeepSeek, GPU retrieval)
│   ├── validation_agent.py        # IC / t-stat evaluation (LightGBM walk-forward)
│   ├── governance_agent.py        # Hard-rule PASS/FAIL filtering
│   ├── diagnosis_agent.py         # Root-cause analysis & repair spec generation
│   └── feature_history.jsonl      # Append-only audit trail (160 KB)
├── phase0_pipeline/               # Generic FAISS index builder (self-contained)
│   ├── build_transcript_index.py  # Earnings call transcripts → FAISS
│   ├── build_theory_index.py      # Academic papers → FAISS
│   └── requirements.txt
├── run_agent.py                   # Main loop orchestration
├── fusion_eval.py                 # Multi-factor fusion evaluation vs M0 baseline
├── demo_agent_visualization.ipynb # Interactive pipeline replay (no GPU needed)
├── data/
│   └── sp500_events.parquet       # S&P 500 earnings event metadata (1.5 MB)
├── agent_output/                  # 3 example features (CSV + report JSON)
├── vector_store/                  # FAISS indices (build via phase0_pipeline)
├── model/                         # BGE-M3 weights (download separately)
└── requirements.txt               # Python dependencies
```

## References

- MASTER: *Multimodal Alpha Research* (AAAI 2024) — market-adaptive multimodal stock selection
- *From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures* — LLM-based text signal extraction
- BGE-M3: *Multi-Lingual, Multi-Granularity Text Embedding* — embedding model used in RAG pipeline
