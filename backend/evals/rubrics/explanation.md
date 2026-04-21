# Explanation Quality Rubric (Suite D)

Used by the LLM-as-judge in `judges/explanation_judge.py`. Score each dimension 1–5.

---

## Dimension 1: Harmonic Accuracy (1–5)

Does the explanation correctly describe the key, chords, and any named progressions?

- **5:** All harmonic claims are correct. Key, mode, and chord function are accurate.
- **4:** Minor imprecision in one claim, but core description is correct.
- **3:** One clearly incorrect claim alongside correct ones.
- **2:** Multiple incorrect claims; explanation is mostly unreliable.
- **1:** Fundamentally wrong about key or chord function.

## Dimension 2: Pedagogical Clarity (1–5)

Would a music beginner understand this explanation?

- **5:** Clear, concrete, no unexplained jargon. Uses analogies or examples.
- **4:** Mostly clear, one unexplained term.
- **3:** Requires some background knowledge, but core idea is accessible.
- **2:** Heavy jargon; a beginner would be lost.
- **1:** Incomprehensible to a non-musician.

## Dimension 3: Appropriate Level (1–5)

Is the depth appropriate for the labeled audience (beginner / intermediate)?

- **5:** Perfectly calibrated. Beginner explanations avoid advanced theory; intermediate explanations go deeper.
- **4:** Slightly over or under the target level.
- **3:** Noticeable mismatch (too simple or too complex for the audience).
- **2:** Clear mismatch throughout.
- **1:** Completely wrong level (e.g. graduate-level theory for a beginner label).

---

## Calibration Notes

Rate 20 examples manually before relying on the judge. Target Cohen's kappa ≥ 0.6 on all dimensions. If kappa < 0.6, tighten the dimension description or add concrete examples to this rubric.
