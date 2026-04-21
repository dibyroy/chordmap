# Design Decisions

Log of non-obvious choices made during the build. Each entry answers: what was the decision, what alternatives were considered, and why this one.

---

## Audio Processing

### librosa chromagram over autochord
- **Decision:** Use librosa CQT chromagram + cosine-similarity template matching for chord detection.
- **Alternatives considered:** autochord (originally chosen), Chordino (VAMP plugin), madmom.
- **Reason:** autochord depends on `vamp`, which requires native VAMP SDK binaries. These fail to build on Windows (`ModuleNotFoundError: No module named 'numpy'` at build time, and no pre-built wheel). The librosa approach uses the same algorithmic idea (chroma template matching) with zero native dependencies. madmom would be more accurate but is heavier and also has build issues on Windows.

### WhisperX for forced alignment
- **Decision:** Use WhisperX instead of plain Whisper.
- **Alternatives considered:** Gentle, aeneas, Montreal Forced Aligner.
- **Reason:** WhisperX adds word-level timestamps on top of Whisper transcription with minimal setup. Gentle requires Java. MFA is overkill for a v1 pipeline.

---

## Eval Harness

### Streamlit dashboard over custom frontend
- **Decision:** Use Streamlit for the eval dashboard, not a React app.
- **Alternatives considered:** Dash, Gradio, custom Next.js page.
- **Reason:** The dashboard is internal tooling used during development. Streamlit is fast to build and doesn't need to be production-quality. Not worth the overhead of a second React app.

### Build eval harness before optimizing
- **Decision:** Phase 3 (eval harness) comes before Phase 5 (iteration loop).
- **Alternatives considered:** Build the app, then eval later.
- **Reason:** Without evals first, iteration is intuition-driven. Changes can't be measured and the writeup loses its credibility.

---

## LLM Layer

### Claude Sonnet 4.6 as primary model
- **Decision:** Default to Sonnet 4.6 for the analyze step.
- **Alternatives considered:** Opus (more capable, slower, more expensive), Haiku (faster, less capable).
- **Reason:** Sonnet hits the right cost/quality tradeoff for a research project. Opus and Haiku are used in Phase 6 cross-model comparison.

### Versioned prompts as text files
- **Decision:** Store prompts in `pipeline/prompts/` as `.txt` files, referenced by name in code.
- **Alternatives considered:** Inline strings, prompt management services.
- **Reason:** Versioned in git, diffable, hashable for system versioning in evals. No external dependency.

---

## Infrastructure

### SQLite + local FS for v1
- **Decision:** No cloud storage or managed database in v1.
- **Alternatives considered:** PostgreSQL, S3, Supabase.
- **Reason:** SQLite is sufficient for a single-user research project. Avoids ops complexity while the pipeline is still changing.

---

_Add new entries here as non-obvious choices come up during the build._
