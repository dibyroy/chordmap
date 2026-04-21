# ChordMap

Chord detection + LLM analysis with rigorous evaluation methodology.

## Overview

ChordMap is a web application that analyzes uploaded songs to detect chords, align them to lyrics (Ultimate Guitar style), and generate musical insights using Claude. It's built as a **vehicle for writing about LLM evaluation**—demonstrating how to measure and iterate on systems that combine music information retrieval with LLM reasoning.

Most chord detection projects treat evaluation as an afterthought. ChordMap puts evals first: measuring not just *what* the system detects, but *when* it fails, *why* it fails, and *how* upstream errors degrade downstream LLM analysis.

## The Project

**What it does:**
1. Upload audio + lyrics
2. Detect chord sequence via chromagram + autochord
3. Align chords to words using forced alignment (WhisperX)
4. Run LLM analysis: key detection, roman numeral analysis, progression identification, beginner-friendly explanation
5. Display synced chord-over-lyrics overlay with diagnostics panel

**Why it exists:**
To demonstrate rigorous eval methodology for AI products. The writeup walks through:
- Building an eval harness *before* you optimize
- Measuring baseline accuracy (Isophonics dataset)
- Testing robustness to realistic errors (degradation curves)
- Calibrating LLM-as-judge for subjective outputs
- Cross-model comparison (Claude Opus/Sonnet/Haiku vs GPT)
- Iteration discipline: hypothesis → change → measured result

## Architecture

```
Frontend (Next.js)
    ↓
Backend (FastAPI) 
    ├─ Stage 1: Audio preprocessing (librosa)
    ├─ Stage 2: Chord detection (autochord/librosa)
    ├─ Stage 3: Lyric alignment (WhisperX)
    ├─ Stage 4: Merge chords + lyrics (custom logic)
    └─ Stage 5: LLM analysis (Anthropic API)
    
Eval Harness (Python)
    ├─ Chord detection accuracy (Isophonics)
    ├─ Key detection (LLM vs Krumhansl baseline)
    ├─ Robustness (synthetic error injection)
    ├─ Explanation quality (LLM-as-judge with calibration)
    └─ Lyric alignment (word-level timing accuracy)
    
Dashboard (Streamlit)
    └─ Score tracking, run comparison, failure analysis
```

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | Next.js + React + Tailwind |
| Audio Processing | librosa, autochord |
| Alignment | WhisperX |
| LLM | Anthropic Claude API (Sonnet 4.6) |
| Storage | SQLite + local filesystem |
| Evals | Python + SQLite + Streamlit |
| Deploy | Vercel (frontend) + Railway/Fly (backend) |

## Build Timeline

- **Week 1:** Backend pipeline end-to-end
- **Week 2:** Frontend upload + audio player
- **Week 3:** Synced chord overlay + deployment
- **Week 4:** Eval harness + dashboard
- **Week 5-6:** Five eval suites + iteration loop
- **Week 7:** Cross-model comparison
- **Week 8:** Writeup + distribution

## Evaluation Suites

### Suite A: Chord Detection (Objective)
- **Dataset:** Isophonics (hand-annotated Beatles, Queen, etc.)
- **Metric:** Chord symbol recall at 3 difficulty levels (root, majmin, majmin7)
- **Baseline:** N/A (evaluates the MIR layer only)

### Suite B: Key Detection (LLM Layer)
- **Dataset:** Ground-truth keys from Isophonics
- **Metric:** Accuracy + credit for relative major/minor
- **Baseline:** Krumhansl-Schmuckler algorithm
- **Finding:** LLM often matches or beats rule-based baseline

### Suite C: Robustness (Most Original)
- **Methodology:** Inject realistic errors into clean chord sequences at rates [0%, 5%, 10%, 15%, 20%, 30%, 50%]
- **Metric:** Degradation curve of downstream LLM analysis quality
- **Output:** "LLM analysis useful up to ~15% chord error, then degrades rapidly"
- **Insight:** Directly answers "how good does my detector need to be?"

### Suite D: Explanation Quality (LLM-as-Judge)
- **Dataset:** 50 chord sequences with audience labels (beginner/intermediate)
- **Rubric:** Accuracy, pedagogical clarity, appropriate level
- **Judge Calibration:** Rate 20 examples yourself, measure Cohen's kappa against judge
- **Finding:** Judge can be reliably calibrated if rubric is specific

### Suite E: Lyric Alignment (Bonus)
- **Dataset:** 10-20 songs hand-labeled with word-level timestamps
- **Metric:** Mean absolute error (ms), per-word error distribution
- **Finding:** WhisperX achieves ~50-100ms MAE on clean vocals

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)
- Anthropic API key

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Run Locally
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000`

### Run Evals
```bash
cd backend
python -m evals.run --suite key_detection --version current
python -m evals.dashboard
```

Visit `http://localhost:8501` for Streamlit dashboard

## Repository Structure

```
chordmap/
├── backend/
│   ├── pipeline/
│   │   ├── preprocess.py       # Audio normalization, resampling
│   │   ├── chords.py            # autochord extraction
│   │   ├── align.py             # WhisperX forced alignment
│   │   ├── merge.py             # Chord-to-lyric alignment
│   │   ├── analyze.py           # Claude API calls
│   │   ├── run.py               # End-to-end runner
│   │   └── prompts/             # Versioned LLM prompts
│   ├── api/
│   │   └── main.py              # FastAPI server
│   ├── evals/
│   │   ├── harness.py           # Runner + data models
│   │   ├── system_version.py    # Version hashing
│   │   ├── datasets/            # Isophonics, custom data
│   │   ├── suites/              # 5 eval implementations
│   │   ├── baselines/           # Krumhansl, etc.
│   │   ├── judges/              # LLM-as-judge code
│   │   ├── rubrics/             # Eval rubrics
│   │   ├── runs/                # Output JSONs
│   │   ├── dashboard.py         # Streamlit app
│   │   └── changelog.md         # Iteration log
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── components/
│   │       ├── AudioPlayer.tsx
│   │       ├── LyricSheet.tsx
│   │       ├── AnalysisSidebar.tsx
│   │       └── Diagnostics.tsx
│   └── package.json
├── DECISIONS.md                 # Design decisions
├── README.md                    # This file
└── .gitignore
```

## Key Design Decisions

See [`DECISIONS.md`](./DECISIONS.md) for rationale on:
- Why autochord over Chordino
- Why WhisperX for alignment
- Why eval harness before iteration
- Why Streamlit for dashboard

## Writeup

The full evaluation narrative and findings will be published as a long-form blog post covering:
1. Why LLMs can't analyze audio directly (architectural corrective)
2. Each eval suite: methodology, baseline, findings
3. Robustness degradation curves and their interpretation
4. LLM-as-judge calibration and its challenges
5. Iteration log: hypothesis → change → measured result
6. Cross-model comparison table
7. Limitations and what's next

**Target:** 3000-5000 words, published to personal blog + Hacker News + ML community channels

## Contributing

This is a portfolio/research project. If you find bugs or have suggestions, please open an issue.

## License

MIT

## Contact

Built by Diby | [@diby](https://twitter.com/diby) | [Blog](https://your-blog.com)
