# Questions for the clinical review (before the final test)

*For the pharmacy team members. Synthetic patients only. Please write your answer in the last column (or reply in our
group chat). The final test is run only after these are answered, so that no answer is changed after seeing the test
results.*

| # | Question | What the system does now | Your decision |
|---|---|---|---|
| 1 | **Severity rule:** should interactions that raise bleeding risk with an anticoagulant be **major** or **moderate**? Examples: warfarin + aspirin, warfarin + ibuprofen, warfarin + fluconazole, warfarin + sulfamethoxazole/trimethoprim, apixaban + ibuprofen | Calls them **moderate**; our expected answers say **major** | major / moderate / depends (please say on what) |
| 2 | **P023** (eGFR 23, metformin 1000 mg twice daily): is this Dosage-section sentence good enough evidence? *"…tablets are contraindicated in patients with an estimated glomerular filtration rate (eGFR) below 30 mL/minute/1.73 m2."* | Flags it as contraindicated, quoting the Dosage section; our expected answer points to the Contraindications section | yes, both sections count / no, only Contraindications |
| 3 | **P013** (CrCl 40 mL/min, digoxin 0.125 mg daily, also on amiodarone): the AI decided the digoxin dose **by itself** is acceptable for this kidney function (it still warns about amiodarone + digoxin) | No separate digoxin dose warning | agree / should also warn about the dose |
| 4 | **Edge case E12** (8 medicines): extra warnings not in our expected answers: **potassium chloride + spironolactone** (moderate, high potassium) and **omeprazole + warfarin** (moderate, INR rise) | Shown as moderate warnings | correct? right severity? |
| 5 | **All 21 edge-case outputs:** open `results/stress_test_rag_v3.csv` in Google Sheets/Excel and fill in the `pharmacist_review` column (OK / not OK + why) | — | — |

If the team decides on a severity rule (question 1), it is added to the system's instructions and checked on the
practice (dev) patients first, then the final test is run once.
