# Final Test Report (Week 4) – Medication Safety Assistant (Track A)

*Generated 2026-10-31 by `python -m src.report` from the files in `results/`. Patients: the 22 locked **test** patients, used once for the final evaluation after all tuning was finished.*

> Not for clinical use. All patients are synthetic.

## 1. What we compared

| System | How it works |
|---|---|
| Baseline 1: keyword checker | Flags two drugs if one's name appears in the other's warning sections; flags a contraindication if a patient condition word appears in a label's Contraindications/Boxed Warning. No AI. |
| Baseline 2: AI only | The AI model reviews the patient from memory, with no label text given. Its quotes are then searched for in the real labels. |
| RAG v1 (Week 2) | Targeted search of the FDA labels (per drug pair and per drug, restricted to the right label and sections, using drug classes) → the AI answers **only** from the retrieved excerpts → citation check removes any warning whose quote is not in the cited excerpt. |
| RAG v2 (Week 3) | v1 + stricter rules (one continuous quote, compare the actual dose with the label limit, pregnancy flagged only if contraindicated/avoid/boxed warning, check every medicine) + patients with more than 40 excerpts reviewed one medicine at a time. |
| RAG v3 (Week 4) | v2 + self-check (the AI states whether each flag is really a problem for this patient; "no" is dropped) + quote repair (a quote that fails the check gets one more request to copy it exactly). |

## 2. Main results

22 dev patients, 18 expected warnings. A warning counts as correct when it names the same drugs as the expected one; **grounded** means it is correct *and* its quoted evidence supports it.

| Score | Baseline 1: keyword checker | Baseline 2: AI only (no search) | RAG v1 (Week 2) | RAG v2 (Week 3 guardrails) | RAG v3 (Week 4 final) |
|---|---|---|---|---|---|
| Precision (warnings given that were correct) | 0.67 | 0.94 | 0.95 | 1.00 | 0.94 |
| Recall (expected warnings found) | 0.78 | 0.94 | 1.00 | 0.83 | 0.94 |
| **F1** | 0.72 | 0.94 | 0.97 | 0.91 | 0.94 |
| **Grounded F1** (correct + supported by its quote) | 0.72 | 0.11 | 0.92 | 0.85 | 0.89 |
| Citation accuracy (correct warnings with supporting evidence) | 100% | 12% | 94% | 93% | 94% |
| Citation accuracy, strict (exact expected sentence/section) | 36% | 6% | 72% | 53% | 59% |
| Severity matches the expected severity | 79% | 82% | 83% | 80% | 88% |
| Safe patients with a false alarm | 3/5 | 0/5 | 1/5 | 0/5 | 1/5 |
| Warnings shown without a verified source | 0/22 | 16/18 | 0/20 | 0/16 | 0/19 |
| Warnings removed by the citation check | 0/22 | 0/18 | 0/20 | 2/18 | 0/19 |

### By warning type (F1; found / expected)

| Type | Baseline 1: keyword checker | Baseline 2: AI only (no search) | RAG v1 (Week 2) | RAG v2 (Week 3 guardrails) | RAG v3 (Week 4 final) |
|---|---|---|---|---|---|
| interaction | 0.96 (11/12, 0 false alarms) | 0.96 (11/12, 0 false alarms) | 0.96 (12/12, 1 false alarm) | 0.96 (11/12, 0 false alarms) | 0.96 (12/12, 1 false alarm) |
| contraindication | 0.46 (3/3, 7 false alarms) | 1.00 (3/3, 0 false alarms) | 1.00 (3/3, 0 false alarms) | 0.80 (2/3, 0 false alarms) | 0.80 (2/3, 0 false alarms) |
| dose | 0.00 (0/3, 0 false alarms) | 0.86 (3/3, 1 false alarm) | 1.00 (3/3, 0 false alarms) | 0.80 (2/3, 0 false alarms) | 1.00 (3/3, 0 false alarms) |

## 4. Errors to learn from

**Baseline 1: keyword checker**

- Missed: P012 aspirin + ticagrelor (interaction); P028 dabigatran (dose); P030 rivaroxaban (dose); P032 dofetilide (dose)
- False alarms: P010 itraconazole (contraindication); P012 ticagrelor (contraindication); P034 digoxin (contraindication); P038 carvedilol (contraindication); P038 rosuvastatin (contraindication); P040 metoprolol (contraindication); P044 metoprolol (contraindication)

**Baseline 2: AI only (no search)**

- Missed: P008 dofetilide + hydrochlorothiazide (interaction)
- False alarms: P008 dofetilide (dose)

**RAG v1 (Week 2)**

- Missed: none
- False alarms: P044 hydrochlorothiazide + metoprolol (interaction)

**RAG v2 (Week 3 guardrails)**

- Missed: P006 diltiazem + simvastatin (interaction); P026 sulfamethoxazole and trimethoprim (contraindication); P030 rivaroxaban (dose)
- False alarms: none

**RAG v3 (Week 4 final)**

- Missed: P024 prasugrel (contraindication)
- False alarms: P044 hydrochlorothiazide + metoprolol (interaction)

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

## 5. AI usage

Free model (Gemini free tier or local Ollama): no cost.

| System | Model | Input tokens | Output tokens | Avg chunks sent |
|---|---|---|---|---|
| Baseline 2: AI only (no search) | gemini:gemini-3.1-flash-lite | 16,563 | 3,769 | – |
| RAG v1 (Week 2) | gemini:gemini-3.1-flash-lite | 205,062 | 5,192 | 24.3 |
| RAG v2 (Week 3 guardrails) | gemini:gemini-3.1-flash-lite | 215,612 | 5,022 | 24.3 |
| RAG v3 (Week 4 final) | gemini:gemini-3.1-flash-lite | 221,092 | 5,322 | 24.3 |

## 6. Method notes

- **No data leakage:** prompts, search settings and the drug-class list are tuned on dev patients only. `src/evaluate.py` refuses to run on the test split without `--final`.
- **Kidney function** (Cockcroft-Gault CrCl, CKD-EPI 2021 eGFR) is calculated in code and given to the AI.
- **Matching:** same patient + same set of drugs. Duplicate warnings count once. Warnings listed in `test_set/acceptable_extras.csv` (e.g. intended dual antiplatelet therapy) are neither rewarded nor penalised.
- **Citation check (guardrail):** a quote must be at least 4 words and appear in the cited excerpt; the RAG system may only cite excerpts it was given. Failing warnings are removed before display.
- **Why F1 and not ROC-AUC:** our systems output a list of warnings, not a probability score per drug pair, so ROC-AUC does not apply. F1 (per warning type), citation accuracy and false alarms on safe patients are the scores that match the clinical task.
- **Reproducible:** every AI answer is saved in `results/cache/`; re-running gives identical results at no cost.

## 7. Reproduce

```
python -m src.chunking
python -m src.index
python -m src.evaluate --retrieval
python -m src.evaluate --system all
python -m src.report
```