# EarningsSignal Agent — LLM-Driven Alpha Factor Discovery Pipeline

An autonomous multi-agent system for discovering and iteratively refining predictive signals from earnings call transcripts. Built for AI Agent engineer roles at quantitative hedge funds.

## What Makes This Different

| Conventional LLM Factor Mining | This Project |
|---|---|
| LLM freely generates factors | **RAG-constrained hypothesis space (6 theoretical clusters × 20 papers)** |
| No quality governance | **Multi-metric hard-rule gates: IC / |t| / zero_ratio / direction_consistency** |
| Trial-and-error without learning | **DiagnosisAgent performs root-cause analysis + generates targeted repair specs** |
| No repair mechanism | **Autonomous closed-loop repair: FAIL → diagnose → fix → re-extract → PASS** |
| Black-box factor production | **Every decision trace recorded in feature_history.jsonl (append-only audit trail)** |

## Architecture

```
Phase 0 (One-time)                    Phase 1 (Agent Loop)
────────────────────────────────────────────────────────────────
                                      ┌──────────────────┐
   Papers (20 PDFs) ──▶ RAG index    │  HypothesisAgent  │◀── repair_spec
                                      │  theory→hypothesis│    (from Diagnosis)
                                      └────────┬─────────┘
                                               │ feature_spec (JSON)
                                      ┌────────▼─────────┐
   Transcript batches ──▶ chunk      │  ExtractionAgent   │
   retrieval (GPU matmul, 947K)      │  LLM batch scoring  │
                                      └────────┬─────────┘
                                               │ feature_df (CSV)
                                      ┌────────▼─────────┐
   sp500_events.parquet ──▶ merge    │  ValidationAgent   │
                                      │  IC · t-stat · NW  │
                                      └────────┬─────────┘
                                               │ val_result
                                      ┌────────▼─────────┐
                                      │ GovernanceAgent    │
                                      │  G1–G4 hard rules  │
                                      └───┬──────────┬────┘
                                    PASS  │          │ FAIL
                                          │   ┌──────▼──────────┐
                                          │   │  DiagnosisAgent   │
                                          │   │  symptom→RAG→LLM  │
                                          │   │  root_cause + fix │
                                          │   └──────┬───────────┘
                                          │          │ _repair_spec
                                          │   ┌──────▼───────────┐
                                          │   │  HypothesisAgent  │
                                          │   │  _repair_queue    │
                                          │   └──────┬───────────┘
                                          │          │ v2 feature
                                          │   ┌──────▼───────────┐
                                          │   │  Extract→Validate │
                                          │   │  →Govern→PASS    │
                                          │   └──────────────────┘
                                   ┌──────▼──────┐
                                   │  fusion_eval │
                                   │  multi-factor│
                                   └──────────────┘
```

## Key Innovations

### 1. Theory-First Hypothesis Generation
HypothesisAgent retrieves from 20 academic papers (BGE-M3 embeddings) across 6 theoretical clusters: `tone_sentiment`, `information_asymmetry`, `forward_guidance`, `qa_subjectivity`, `managerial_behavior`, `alpha_discovery`. The LLM reads theory FIRST, then derives a feature spec — never the reverse.

### 2. Multi-Metric Governance with Zero-Type Classification
GovernanceAgent enforces hard rules with adaptive thresholds:
- **G1**: OOS coverage ≥ 5%
- **G2**: zero_ratio — `uniform ≤ 70%` (legal zeros), `systemic_sparse` (triggers repair), `concentrated ≤ 45%` (evasive zeros)
- **G3**: |t-stat| ≥ 1.5 (NW lags=4)
- **G4**: direction_consistency ≥ 60%

### 3. Autonomous Diagnosis & Repair Closed-Loop
When a feature fails with strong signal but high zero_ratio (G2-only), DiagnosisAgent:
1. Maps failure symptoms → RAG queries (rule-based, no LLM hallucination)
2. Searches theory index for relevant evidence
3. Produces structured diagnosis: `{root_cause, fix, avoid, rag_refs}`
4. Salvage check: IC > 0.08, |t| > 2.0, only_G2 → generates repair spec (v2 extraction_instruction)
5. HypothesisAgent prioritizes repair specs in `_repair_queue` over new generation
6. v2 feature is re-extracted and re-validated at full scale

**Verified repair closed-loop** (`contrastive_connectives`, 2,462 episodes):

| Metric | v1 (original) | v2 (repaired) | Change |
|--------|--------------|---------------|--------|
| IC | +0.1361 | +0.1350 | −0.0011 (99.2% retained) |
| \|t-stat\| | 2.858 | 3.155 | **+10.4%** |
| zero_ratio | 78.6% | 44.5% | **−34.1pp (−43%)** |
| direction | 73% | 73% | unchanged |
| Governance | FAIL (G2) | **PASS** | |

The LLM diagnosed "discretization thresholds too rigid" and generated a v2 instruction using continuous semantic scoring instead of word-count bins — preserving signal while drastically reducing zeros.

### 4. Full Pipeline Traceability
- `feature_history.jsonl`: append-only audit trail of all 55 explored features
- `agent_output/trace_*.json`: structured per-step trace (73 steps, 92% tool success rate)
- `agent_output/loop_checkpoint.json`: FSM checkpoint for crash recovery
- `viz_demo/agent_pipeline_demo.html`: self-contained HTML visualization of the complete pipeline

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add your SiliconFlow API key

# 3. Build FAISS indices (one-time, see phase0_pipeline/README.md)
cd phase0_pipeline
pip install -r requirements.txt
python build_transcript_index.py    # ~947K chunks, ~3.9GB
python build_theory_index.py         # 20 papers

# 4. Run the Agent loop
cd ..
python run_agent.py --max-iter 20

# 5. Evaluate multi-factor fusion
python fusion_eval.py

# 6. Visualize pipeline (no GPU needed)
python viz_demo/build_demo.py
# Open viz_demo/agent_pipeline_demo.html in browser
```

## Exploration Results (S&P 500, 55 features explored)

### Top Discovered Features

| Feature | IC | |t-stat| | zero_ratio | PASS | Notes |
|---------|-----|---------|------------|------|-------|
| qa_spontaneity | +0.1656 | 4.983 | 30.6% | ✅ | Q&A spontaneity vs prepared remarks |
| moderate_positive_tone | +0.1523 | 3.742 | 44.4% | ✅ | Non-extreme positive emotion ratio |
| contrastive_connectives | +0.1361 | 2.858 | 78.6% | ❌→✅ | **Repaired**: v2 zr=44.5%, PASS |
| managerial_certainty_tone | +0.1028 | 2.856 | 62.4% | ❌ | G2-only, salvageable candidate |
| mgmt_tone_confidence | +0.0727 | 2.637 | 3.8% | ✅ | Strong signal, almost zero zero_ratio |

### Fusion Performance (BASE13 financial + 7 text features)

| Period | M0 (BASE13) | + Text | Δ | Regime |
|--------|-------------|--------|---|--------|
| 2020 (val) | +0.0673 | **+0.0828** | **+23%** | COVID: text captures uncertainty |
| 2022 (test) | +0.1911 | **+0.2009** | +5.1% | Bear market: tone shift signal |
| 2023 (test) | +0.0840 | **+0.1007** | +19.9% | AI rally: narrative divergence |

| Risk Metric | M0 (BASE13) | + Text | Improvement |
|-------------|-------------|--------|-------------|
| Long-Short IR | 2.484 | **2.733** | **+10.0%** |
| Max Drawdown | −1.37% | **−0.49%** | **−64%** |

### Residual IC — Orthogonal to Financial Features

- `forward_numeric_specificity`: residual IC = **+0.108** (3× avg BASE13 marginal IC)
- `qa_spontaneity`: residual IC = **+0.043**
- `guidance_revision_direction` (v3): residual IC = **+0.033**

## Project Structure

```
Fullproject/
├── agent_core/                     # Five-layer agent pipeline
│   ├── config.py                   # Shared paths, API keys, constants
│   ├── hypothesis_agent.py         # Theory-First RAG hypothesis generation
│   ├── extraction_agent.py         # GPU retrieval + LLM batch scoring
│   ├── validation_agent.py         # LightGBM walk-forward IC evaluation
│   ├── governance_agent.py         # Hard-rule PASS/FAIL filtering
│   ├── diagnosis_agent.py          # Root-cause analysis + repair spec generation
│   └── feature_history.jsonl       # Append-only audit trail (55 features)
├── harness/                        # Agent runtime framework
│   ├── loop.py                     # 9-state FSM engine + checkpoint/resume
│   ├── guardrail.py                # Pluggable 5-gate pipeline
│   ├── adapters.py                 # Handler factories (7 states)
│   ├── tools.py                    # ToolRegistry (timeout/retry/fallback)
│   ├── context.py                  # ContextConstructor (3-level assembly)
│   ├── tracer.py                   # Structured trace + replay
│   └── memory/
│       └── episodic.py             # Cross-session BGE-M3 episodic memory
├── phase0_pipeline/                # FAISS index builder (self-contained)
│   ├── build_transcript_index.py
│   ├── build_theory_index.py
│   └── README.md
├── viz_demo/                       # Pipeline visualization
│   ├── build_demo.py               # HTML generator from history data
│   └── agent_pipeline_demo.html    # Self-contained 6-stage demo
├── tests/
│   └── test_harness.py             # Harness component tests
├── run_agent.py                    # Main loop (5-layer orchestration)
├── run_harness.py                  # Harness-based loop (FSM engine)
├── fusion_eval.py                  # Multi-factor fusion vs M0 baseline
├── demo_repair_closed_loop.py      # Closed-loop repair analysis tool
├── live_repair_test.py             # End-to-end repair live test
├── fullscale_repair_test.py        # Full-scale v1 vs v2 validation
├── data/
│   └── sp500_events.parquet        # Earnings event metadata (1.5 MB)
├── agent_output/                   # 4 example features (CSV + report)
├── vector_store/                   # FAISS indices (build via phase0_pipeline)
├── model/                          # BGE-M3 weights (download separately)
├── 设计日志.md                      # Design decision log (Chinese)
├── .env.example                    # Environment template
├── requirements.txt
└── README.md
```

## Limitations

- **Coverage**: ~69% of S&P 500 episodes have transcript data; uncovered samples rely on BASE13 financial features
- **LLM API cost**: ~10M tokens per full extraction run (DeepSeek V4 Flash, ~$0.14/1M input tokens)
- **Repair depth**: Limited to 1 level (v1→v2) to prevent infinite cascade
- **Coverage bottleneck**: 80% of FAILs are G1_coverage (retrieval scope), not signal quality — identified as next optimization target

## References

- *MASTER: Market-Adaptive Multimodal Stock Selection* (AAAI 2024)
- *From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures*
- *CogAlpha: Cognitive Alpha Mining via LLM-Driven Code-Based Evolution* (arXiv 2511.18850)
- *AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration* (SIGKDD 2025)
- BGE-M3: *Multi-Lingual, Multi-Granularity Text Embedding*
