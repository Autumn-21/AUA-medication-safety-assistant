## 4b. What the final test shows

*The 22 test patients (18 expected warnings, 5 safe patients) were never used while building or tuning the system.
The AI answers for the test patients were generated in one run on 30 Oct 2026 with the frozen RAG v3 code and saved in
`results/cache/`; the clinical review (31 Oct) asked for no changes, so these answers are final and were scored as
they are. Same free model as for dev: gemini-3.1-flash-lite.*

**Dev vs test (our final system, RAG v3)**

| | Dev (practice, used for tuning) | Test (locked, final) |
|---|---|---|
| F1 | 1.00 | **0.94** |
| Grounded F1 | 0.95 | **0.89** |
| Expected warnings found | 19 of 19 | 17 of 18 |
| Safe patients with a false alarm | 0 of 5 | 1 of 5 |
| Warnings shown without a verified source | 0 of 19 | **0 of 19** |
| Severity matches | 84% | 88% |

As expected, the test score is a little lower than dev: the Week 3–4 rules were written after seeing dev errors, so dev
was optimistic. The test numbers are the honest ones to present.

**Main findings**

1. **Grounding is what separates the systems.** The AI-only baseline finds almost as many problems (F1 0.94) but only
   2 of its 18 warnings come with a quote that really exists in the label (grounded F1 0.11; 16 warnings shown with
   no verified source). All three RAG versions are grounded (grounded F1 0.85–0.92) and **never showed a warning
   without a verified label quote**. The keyword baseline is grounded but gives false alarms on 3 of 5 safe patients.
2. **The three RAG versions are close on test.** v1 F1 0.97 (1 false alarm), v2 0.91 (0 false alarms, 3 misses),
   v3 0.94 (1 miss, 1 false alarm). With 18 warnings, one warning moves F1 by about 0.03, and the free model's answers
   vary between runs, so these differences are within normal variation. We do **not** claim that v3 beats v1.
3. **What the Week 3–4 guardrails changed on test:** the v2 citation check removed two correct warnings because the
   AI's quotes did not match the label (P006 simvastatin with diltiazem; P030 rivaroxaban dose), so v2 missed them.
   In v3 the **quote repair** rescued P006 and P030 was quoted correctly, so v3 recovered both (recall 0.83 → 0.94).
   Severity agreement rose from 80% (v2) to 88% (v3).

**Errors of the final system (RAG v3) on test**

| Patient | Error | Likely cause | Next step |
|---|---|---|---|
| P024 (prasugrel, TIA 2 years ago) | **Missed contraindication.** v1 and v2 found it; v3 returned no warning at all | Possibly the self-check: "2 years ago" may have been read as no longer relevant, although the label contraindicates prasugrel after any prior TIA or stroke. Not dismissed by the code: the AI simply did not raise it | Prompt rule: a past history of a contraindicated condition still counts; re-test on new patients |
| P044 (safe: metoprolol + hydrochlorothiazide) | **False alarm (moderate):** "additive blood-pressure lowering" | A normal, intended combination; the label sentence is real but not a safety problem. v1 made the same error, v2 did not | Pharmacist list of intended combinations (like the dual antiplatelet rule) |
| P002, P018 (warfarin + fluconazole / naproxen) | Severity "moderate" instead of "major" | Same pattern as dev; the clinical review kept our expected answers | Accepted as a known limitation |

**Limitations to state in the pitch**

- Small test set (22 synthetic patients, 18 warnings): results show the approach works, not clinical-grade accuracy.
- One run per system with a free, small model; answers vary a little between runs.
- 63 cardiovascular-related medicines only; anything else is listed as "not checked".
- A prototype for a course. Not for clinical use; a pharmacist must review every output.
