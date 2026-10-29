# Practice patients (test set)

**All patients are synthetic (made up). No real patient data.**
**Status: draft. Every row must be checked by a pharmacy team member before use.**

## Files

| File | What it is |
|---|---|
| `patients.csv` | 44 practice patients: age, sex, weight, creatinine, kidney function, pregnancy, conditions, allergies, medicines |
| `expected_flags.csv` | 37 correct warnings: what the system SHOULD find for each patient, with the label source |
| `build_test_set.py` | The script that makes the two CSV files. Edit the cases here, then run it again |
| `verify_test_set.py` | Checks that every expected warning is really written in the FDA label text |

## What is inside

| Category | Patients | What it tests |
|---|---|---|
| interaction | 20 | Two medicines that are risky together |
| contraindication | 6 | A medicine unsafe for this patient (pregnancy, asthma, allergy, kidney, prior stroke) |
| dose | 6 | A dose that is wrong for kidney function, age or weight |
| multiple | 2 | Patients with 2-3 problems at once |
| negative | 10 | **Safe patients with no warning expected**, including "traps" that look risky but are fine |

Each patient is marked `dev` or `test` (half each, every category in both):
- **dev**: use freely while building and improving the system.
- **test**: **locked until Week 4.** Do not look at the system's results on these
  while tuning prompts. This keeps the final score honest.

## Column guide

- `crcl_ml_min`: creatinine clearance, Cockcroft-Gault, actual body weight (calculated in code).
- `egfr_ml_min_1_73m2`: eGFR, CKD-EPI 2021 equation (calculated in code).
- `medications`: `drug | dose | frequency`, separated by `;`.
- `severity`: `contraindicated` (label says contraindicated / do not use), `major` (avoid, or change the dose),
  `moderate` (monitor).
- `source_drug` + `source_section` + `evidence_phrase`: where the correct answer is written in
  `data/label_sections.csv`. This is the "correct citation" used to score citation accuracy.
- `notes`: why the case exists, or what extra warnings are acceptable.

## How the scores work

For each patient, compare the system's warnings with `expected_flags.csv`:
- **Correct (true positive):** the system flags the same drugs and the same type of problem.
- **Missed (false negative):** an expected warning the system did not give.
- **False alarm (false positive):** a warning that is not in the expected list
  (notes may list extra warnings that are acceptable and not counted).
- **Precision** = correct / all warnings given. **Recall** = correct / all expected warnings.
- **Citation accuracy** = how many correct warnings cite the right label section.

## Pharmacist review checklist (per patient)

1. Is the patient realistic (age, weight, creatinine, conditions)?
2. Is every expected warning clinically correct, with the right severity?
3. Is anything missing that a pharmacist would flag? Add it to `expected_flags.csv`.
4. For negative patients: is it truly safe?
5. Change `review_status` to `reviewed by <name>`.

## After re-downloading the labels

Run this from the project folder:

```
python test_set/verify_test_set.py
```

It should print `37/37 expected flags are backed by the label text.` If a row fails, the label
wording changed: find the new sentence in `data/label_sections.csv` and update `evidence_phrase`.

## Adding more patients (goal: 60-100)

Add rows to the `P` list (patients) and `F` list (expected warnings) in `build_test_set.py`,
then run `python build_test_set.py` inside `test_set/`, then run the verify script.
