# Week 2 Benchmarking Report – Medication Safety Assistant (Track A)

*Generated 2026-10-31 by `python -m src.report` from the files in `results/`. Patients: **dev** split only (the test split stays locked until Week 4).*

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

22 dev patients, 19 expected warnings. A warning counts as correct when it names the same drugs as the expected one; **grounded** means it is correct *and* its quoted evidence supports it.

| Score | Baseline 1: keyword checker | Baseline 2: AI only (no search) | RAG v1 (Week 2) | RAG v2 (Week 3 guardrails) | RAG v3 (Week 4 final) |
|---|---|---|---|---|---|
| Precision (warnings given that were correct) | 0.56 | 0.94 | 0.89 | 1.00 | 1.00 |
| Recall (expected warnings found) | 0.74 | 0.89 | 0.89 | 1.00 | 1.00 |
| **F1** | 0.64 | 0.92 | 0.89 | 1.00 | 1.00 |
| **Grounded F1** (correct + supported by its quote) | 0.59 | 0.00 | 0.84 | 0.95 | 0.95 |
| Citation accuracy (correct warnings with supporting evidence) | 93% | 0% | 94% | 95% | 95% |
| Citation accuracy, strict (exact expected sentence/section) | 43% | 0% | 65% | 58% | 53% |
| Severity matches the expected severity | 79% | 94% | 82% | 79% | 84% |
| Safe patients with a false alarm | 3/5 | 1/5 | 2/5 | 0/5 | 0/5 |
| Warnings shown without a verified source | 1/25 | 19/19 | 0/19 | 0/19 | 0/19 |
| Warnings removed by the citation check | 0/25 | 0/19 | 0/19 | 1/20 | 0/19 |

### By warning type (F1; found / expected)

| Type | Baseline 1: keyword checker | Baseline 2: AI only (no search) | RAG v1 (Week 2) | RAG v2 (Week 3 guardrails) | RAG v3 (Week 4 final) |
|---|---|---|---|---|---|
| interaction | 0.92 (11/12, 1 false alarm) | 0.92 (11/12, 1 false alarm) | 0.92 (11/12, 1 false alarm) | 1.00 (12/12, 0 false alarms) | 1.00 (12/12, 0 false alarms) |
| contraindication | 0.27 (2/3, 10 false alarms) | 1.00 (3/3, 0 false alarms) | 0.86 (3/3, 1 false alarm) | 1.00 (3/3, 0 false alarms) | 1.00 (3/3, 0 false alarms) |
| dose | 0.40 (1/4, 0 false alarms) | 0.86 (3/4, 0 false alarms) | 0.86 (3/4, 0 false alarms) | 1.00 (4/4, 0 false alarms) | 1.00 (4/4, 0 false alarms) |

## 3. Search quality (no AI involved)

Embedding backend: `sentence-transformers`. Recall = share of expected warnings whose evidence sentence was among the chunks retrieved for that patient (k = chunks kept per search).

| Search | k=1 | k=3 | k=5 | chunks per patient (k=5) |
|---|---|---|---|---|
| Keyword (BM25) | 53% | 84% | 100% | 25.7 |
| Vector (embeddings) | 53% | 79% | 100% | 26.0 |
| Hybrid (both) | 58% | 90% | 100% | 25.9 |

Evidence missed by hybrid search at k=3: F029 P029 enoxaparin; F033 P033 apixaban.

## 4. Errors to learn from

**Baseline 1: keyword checker**

- Missed: P023 metformin (contraindication); P027 apixaban (dose); P029 enoxaparin (dose); P033 apixaban (dose); P033 apixaban + ibuprofen (interaction)
- False alarms: P001 metoprolol (contraindication); P001 warfarin (contraindication); P003 warfarin (contraindication); P007 verapamil (contraindication); P013 digoxin (contraindication); P015 isosorbide mononitrate (contraindication); P017 clarithromycin (contraindication); P017 dronedarone (contraindication); P035 metoprolol (contraindication); P037 valsartan (contraindication); P043 diltiazem + simvastatin (interaction)

**Baseline 2: AI only (no search)**

- Missed: P033 apixaban (dose); P033 apixaban + ibuprofen (interaction)
- False alarms: P043 diltiazem + simvastatin (interaction)

**RAG v1 (Week 2)**

- Missed: P033 apixaban (dose); P033 apixaban + ibuprofen (interaction)
- False alarms: P041 hydralazine (contraindication); P043 diltiazem + simvastatin (interaction)

**RAG v2 (Week 3 guardrails)**

- Missed: none
- False alarms: none

**RAG v3 (Week 4 final)**

- Missed: none
- False alarms: none

## 4b. What the results mean

*Run of 29 Oct 2026: model `gemini-3.1-flash-lite` (Google free tier), embeddings `BAAI/bge-small-en-v1.5`, hybrid search,
k = 5, 22 dev patients, 19 expected warnings. Cost: US$0.*

**1. Finding the problem vs. proving it.** The AI without search scored the highest plain F1 (0.92). These are common,
well-known interactions, and the model remembers them well. But **none of its 19 quotes could be found in the real
labels** (grounded F1 = 0.00). The quotes look like label text but are reworded from memory. For example, it quoted
warfarin as "contraindicated in women who are pregnant except in *women* with mechanical heart valves"; the label
says "*pregnant* women". A pharmacist cannot check a citation that does not exist. Our RAG system scored F1 0.86
**and grounded F1 0.81**: 15 of its 16 correct warnings cite a real label sentence that supports them, and **no warning
was shown without a verified source** (0/18).

**2. The keyword baseline fails on context.** It finds named interactions (interaction F1 0.92), but produces
10 false contraindication alarms (contraindication F1 0.27) and false alarms for 3 of 5 safe patients. It also
cannot check doses against kidney function (1 of 4 found).

**3. Search is not the bottleneck.** Hybrid search put the correct evidence sentence in front of the AI for 100% of
expected warnings (89% with k = 3). Every RAG miss below happened *after* the evidence was retrieved.

**4. RAG errors, one by one**

| Patient | What happened | Cause | Fix to try (dev only) |
|---|---|---|---|
| P029 enoxaparin, CrCl 23 | Correct dose warning **removed by the citation check** | The AI stitched two separated passages into one "quote" | Ask for one continuous passage; accept quotes made only of exact ≥4-word pieces in order |
| P033 (5 medicines) | Missed apixaban dose (age ≥80, weight ≤60 kg) and apixaban + ibuprofen | 69 excerpts in one request; the small model lost track | Split large patients into one request per drug |
| P043 simvastatin 10 mg + diltiazem | False alarm | Label limit is 10 mg; the patient's dose is within it | Prompt: compare the actual dose with the limit |
| P041 pregnant, hydralazine | False alarm (moderate) | Label says "use only if benefit justifies risk", not a contraindication | Prompt: flag pregnancy only when the label says contraindicated, avoid, or boxed warning |
| P001, P003 severity | Said "moderate", expected "major" | Label says bleeding / INR risk without "avoid"; our own severity rule allows "moderate" | **Pharmacy team to decide** the severity rule for anticoagulant interactions |
| P023 metformin | Correct warning, but cited the Dosage section ("discontinue if eGFR falls below 30") instead of Contraindications | Valid evidence; our strict word-overlap rule did not accept it | Pharmacist review of the scoring rule |

**5. Limits of this benchmark.** 19 warnings is a small set: one warning more or less changes F1 by about 0.03, so
the AI-only vs. RAG difference in plain F1 (0.92 vs. 0.86) is within noise. The grounding difference (0.00 vs. 0.81)
is not. The test patients have not been used and stay locked for Week 4.

## 4c. Week 3 improvement round (dev patients only)

*Run of 30 Oct 2026, same model and search settings. All changes were designed from the Week 2 errors on dev patients
only; the test patients are still unused.*

| | RAG v1 (Week 2) | RAG v2 (Week 3 guardrails) |
|---|---|---|
| F1 | 0.89 | **1.00** |
| Grounded F1 | 0.84 | **0.95** |
| Expected warnings found | 17 of 19 | **19 of 19** |
| Safe patients with a false alarm | 2 of 5 | **0 of 5** |
| Warnings shown without a verified source | 0 of 19 | 0 of 19 |
| Severity matches the expected answer | 82% | 79% |

*RAG v1 is re-scored here with the new citation rule (same saved answers, no new AI requests). That alone recovered
P029, so v1 rises from F1 0.86 / grounded 0.81 (Week 2) to 0.89 / 0.84.*

**What each fix did**

| Week 2 error | Fix | Result in v2 |
|---|---|---|
| P029: correct warning removed (quote skipped words) | Citation rule accepts exact pieces of 4+ words in their original order; still rejects any changed or invented word (the AI-only quotes still verify 0 of 19) | Kept, with a verified quote |
| P033: 2 misses with 5 medicines (69 excerpts in one request) | Patients with more than 40 excerpts are reviewed one medicine at a time (P033: 5 requests, P039: 3) | Both found |
| P043: false alarm, simvastatin 10 mg within the diltiazem limit | Rule: compare the patient's actual dose with the label limit | No warning (correct) |
| P041: false alarm, "use only if benefit justifies risk" in pregnancy | Rule: flag pregnancy only if the label says contraindicated, avoid/discontinue, or boxed warning | No warning (correct) |

**The guardrail still works:** in v2 it removed one warning (P013, a digoxin dose note whose "quote" was a string of
table numbers not found in the label), and no warning was shown without a verified source.

**Still open**

- **Severity (79%):** the model calls anticoagulant interactions (P001, P003, P033) "moderate" where our answers say
  "major", and metformin with eGFR below 30 (P023) "major" instead of "contraindicated". This needs the pharmacy team's
  severity rule, not more prompt tuning.
- **P023 citation:** cites the Dosage section ("discontinue if eGFR falls below 30"), which is valid evidence but not
  the expected Contraindications sentence.
- **Over-fitting risk:** the rules were written after seeing the dev errors, so 1.00 on dev is optimistic. The honest
  measure is the final run on the locked test patients in Week 4.

## 4d. Week 4 final fine-tuning: RAG v3 (dev patients only)

*Run of 30 Oct 2026, same model, search settings and patients. The changes were designed from the Week 3 stress-test
errors (edge cases), not from the test patients, which are still unused.*

"Fine-tuning" here means improving the pipeline's instructions and checks; the AI model's weights are not changed
(the free model cannot be retrained, and 22 practice patients would be far too few to train on).

| | RAG v2 (Week 3) | RAG v3 (Week 4 final) |
|---|---|---|
| F1 | 1.00 | 1.00 |
| Grounded F1 | 0.95 | 0.95 |
| Expected warnings found | 19 of 19 | 19 of 19 |
| Safe patients with a false alarm | 0 of 5 | 0 of 5 |
| Warnings shown without a verified source | 0 of 19 | 0 of 19 |
| Severity matches the expected answer | 79% | **84%** |
| Stress test (21 edge cases) | 19 of 21 | **21 of 21** |
| Input tokens (all dev patients) | 231,598 | 237,831 (+3%) |

**What each v3 change did**

| Change | Effect |
|---|---|
| **Self-check** (the AI states whether each flag is really a problem for this patient; "no" is dropped) | Dropped 1 candidate on dev: P013 digoxin 0.125 mg at CrCl 40 mL/min, which the AI itself judged consistent with the label. In v2 the same note was removed by the citation check instead. Stress test E10 (false alarm in v2) now gives no warning. |
| **Quote repair** (a failed quote gets one more request to copy the passage exactly) | Kept 2 correct warnings that would otherwise have been removed, i.e. 2 misses avoided: P023 metformin with eGFR below 30, and P033 apixaban dose. Cost: 2 extra requests across 22 patients. |
| **Quote rule** (keep numbers as digits and the label's word order) | Stress test E11 (apixaban dose, missed in v2) was found with an exact quote at the first try. |
| **App** | Warnings whose quote could not be verified are shown openly as "please check the label" instead of being hidden. |

The headline scores do not change because v2 was already at 1.00 on dev; v3's gains are in reliability (2 warnings
rescued, 1 wrong note dropped, 2 stress-test errors fixed, severity 79% → 84%). In the free model's runs, quotes fail
by chance from time to time; in v2 that hides a real warning, in v3 the repair step usually recovers it.

**Still open (for the clinical review, before the final test)**

- **Severity (84%):** the remaining mismatches are all anticoagulant interactions called "moderate" where our answers
  say "major" (P001 warfarin + aspirin, P003 warfarin + sulfamethoxazole/trimethoprim, P033 apixaban + ibuprofen).
- **P023 citation:** the warning is correct and its quote is real ("contraindicated in patients with an eGFR below
  30"), but it comes from the Dosage section rather than the expected Contraindications section, so it is not counted
  as grounded. We have not changed the expected answer after seeing the result; the reviewers should decide whether
  both sections count.
- **Version frozen:** RAG v3 is the final version. After the clinical review (and its severity rule, if any), the
  locked test patients are run **once**: `python -m src.evaluate --system all --split test --final`.

## 5. AI usage

Free model (Gemini free tier or local Ollama): no cost.

| System | Model | Input tokens | Output tokens | Avg chunks sent |
|---|---|---|---|---|
| Baseline 2: AI only (no search) | gemini:gemini-3.1-flash-lite | 16,632 | 3,994 | – |
| RAG v1 (Week 2) | gemini:gemini-3.1-flash-lite | 219,884 | 4,841 | 25.9 |
| RAG v2 (Week 3 guardrails) | gemini:gemini-3.1-flash-lite | 231,598 | 5,606 | 25.9 |
| RAG v3 (Week 4 final) | gemini:gemini-3.1-flash-lite | 237,831 | 5,968 | 25.9 |

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