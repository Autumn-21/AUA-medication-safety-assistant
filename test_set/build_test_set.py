"""Builds the Track A practice-patient test set (synthetic patients only).

Writes:
  patients.csv        one row per synthetic patient
  expected_flags.csv  one row per warning the system SHOULD raise (gold answers)
"""
import csv
import math

# ---------------------------------------------------------------------------
# Kidney function is calculated in code, never by the AI.
# ---------------------------------------------------------------------------
def crcl_cockcroft_gault(age, weight_kg, scr, sex):
    """Creatinine clearance (mL/min), actual body weight."""
    v = (140 - age) * weight_kg / (72 * scr)
    return v * 0.85 if sex == "F" else v


def egfr_ckd_epi_2021(age, scr, sex):
    """eGFR (mL/min/1.73m2), CKD-EPI 2021 race-free equation."""
    k, a = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    v = 142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200 * 0.9938 ** age
    return v * 1.012 if sex == "F" else v


# ---------------------------------------------------------------------------
# Patients: (id, category, age, sex, weight, SCr, pregnant, conditions,
#            allergies, medications, notes)
# Medications: "drug | dose | frequency" separated by ";"
# ---------------------------------------------------------------------------
P = [
    # --- Drug-drug interactions --------------------------------------------
    ("P001", "interaction", 72, "M", 80, 1.0, "No", "atrial fibrillation; hypertension", "none",
     "warfarin | 5 mg | once daily; aspirin | 81 mg | once daily; metoprolol succinate | 50 mg | once daily",
     "No stent or other reason for aspirin documented."),
    ("P002", "interaction", 68, "F", 65, 0.9, "No", "atrial fibrillation; oesophageal candidiasis", "none",
     "warfarin | 5 mg | once daily; fluconazole | 200 mg | once daily for 14 days", ""),
    ("P003", "interaction", 75, "M", 78, 1.1, "No", "history of DVT; urinary tract infection", "none",
     "warfarin | 4 mg | once daily; sulfamethoxazole and trimethoprim | 800/160 mg | twice daily",
     "Label calls SMX-TMP 'cotrimoxazole': tests synonym handling."),
    ("P004", "interaction", 64, "M", 90, 1.0, "No", "hyperlipidaemia; community-acquired pneumonia", "none",
     "simvastatin | 40 mg | once daily; clarithromycin | 500 mg | twice daily", ""),
    ("P005", "interaction", 70, "M", 82, 1.1, "No", "atrial fibrillation; hyperlipidaemia", "none",
     "amiodarone | 200 mg | once daily; simvastatin | 40 mg | once daily", ""),
    ("P006", "interaction", 66, "F", 70, 0.9, "No", "hypertension; stable angina; hyperlipidaemia", "none",
     "diltiazem ER | 240 mg | once daily; simvastatin | 20 mg | once daily", ""),
    ("P007", "interaction", 69, "M", 85, 1.0, "No", "atrial fibrillation; hypertension", "none",
     "dofetilide | 500 mcg | twice daily; verapamil ER | 240 mg | once daily", ""),
    ("P008", "interaction", 71, "F", 68, 0.9, "No", "atrial fibrillation; hypertension; ankle oedema", "none",
     "dofetilide | 500 mcg | twice daily; hydrochlorothiazide | 25 mg | once daily", ""),
    ("P009", "interaction", 62, "M", 88, 1.0, "No", "coronary stent placed 2 months ago; GERD", "none",
     "clopidogrel | 75 mg | once daily; aspirin | 81 mg | once daily; omeprazole | 20 mg | once daily",
     "Clopidogrel + aspirin is intended dual antiplatelet therapy after a stent: not an error."),
    ("P010", "interaction", 58, "F", 72, 0.8, "No", "atrial fibrillation; fungal nail infection", "none",
     "apixaban | 5 mg | twice daily; itraconazole | 200 mg | once daily", "Label names a drug class, not itraconazole directly."),
    ("P011", "interaction", 55, "M", 95, 0.9, "No", "deep vein thrombosis (treatment); tuberculosis", "none",
     "rivaroxaban | 20 mg | once daily; rifampin | 600 mg | once daily", ""),
    ("P012", "interaction", 63, "M", 84, 1.0, "No", "acute coronary syndrome", "none",
     "ticagrelor | 90 mg | twice daily; aspirin | 325 mg | once daily", ""),
    ("P013", "interaction", 78, "F", 60, 1.1, "No", "heart failure; atrial fibrillation", "none",
     "digoxin | 0.125 mg | once daily; amiodarone | 200 mg | once daily (started this week)", ""),
    ("P014", "interaction", 74, "M", 76, 1.3, "No", "heart failure", "none",
     "spironolactone | 25 mg | once daily; potassium chloride ER | 20 mEq | once daily; furosemide | 40 mg | once daily", ""),
    ("P015", "interaction", 60, "M", 85, 1.0, "No", "stable angina; erectile dysfunction", "none",
     "isosorbide mononitrate ER | 60 mg | once daily; sildenafil | 50 mg | as needed", ""),
    ("P016", "interaction", 67, "F", 70, 0.9, "No", "chronic angina; acute sinusitis", "none",
     "ranolazine | 500 mg | twice daily; clarithromycin | 500 mg | twice daily", ""),
    ("P017", "interaction", 70, "M", 80, 1.0, "No", "paroxysmal atrial fibrillation; bronchitis", "none",
     "dronedarone | 400 mg | twice daily; clarithromycin | 500 mg | twice daily", ""),
    ("P018", "interaction", 73, "F", 64, 0.9, "No", "atrial fibrillation; osteoarthritis", "none",
     "warfarin | 3 mg | once daily; naproxen | 500 mg | twice daily", ""),
    ("P019", "interaction", 61, "M", 92, 1.0, "No", "atrial fibrillation; epilepsy", "none",
     "apixaban | 5 mg | twice daily; carbamazepine | 400 mg | twice daily", ""),
    ("P020", "interaction", 69, "M", 79, 1.2, "No", "hypertension", "none",
     "enalapril | 10 mg | twice daily; potassium chloride ER | 20 mEq | once daily", "Moderate: monitor potassium."),
    # --- Contraindications (drug-patient) ----------------------------------
    ("P021", "contraindication", 29, "F", 62, 0.7, "Yes (12 weeks)", "history of DVT; no mechanical heart valve", "none",
     "warfarin | 5 mg | once daily", ""),
    ("P022", "contraindication", 33, "F", 70, 0.7, "Yes (20 weeks)", "chronic hypertension", "none",
     "losartan | 50 mg | once daily", ""),
    ("P023", "contraindication", 76, "F", 58, 1.9, "No", "type 2 diabetes; chronic kidney disease", "none",
     "metformin | 1000 mg | twice daily; atorvastatin | 20 mg | once daily", "eGFR is below 30."),
    ("P024", "contraindication", 66, "M", 80, 1.0, "No", "acute coronary syndrome with stent; transient ischaemic attack 2 years ago", "none",
     "prasugrel | 10 mg | once daily; aspirin | 81 mg | once daily", ""),
    ("P025", "contraindication", 45, "M", 82, 0.9, "No", "hypertension; bronchial asthma", "none",
     "propranolol | 40 mg | twice daily; albuterol inhaler | 2 puffs | as needed",
     "Albuterol is outside the drug set: the system should say it has no label for it, not invent one."),
    ("P026", "contraindication", 59, "F", 68, 0.8, "No", "urinary tract infection", "sulfonamide antibiotics (rash)",
     "sulfamethoxazole and trimethoprim | 800/160 mg | twice daily", ""),
    # --- Kidney / patient-based dose adjustment ----------------------------
    ("P027", "dose", 82, "F", 55, 1.6, "No", "atrial fibrillation", "none",
     "apixaban | 5 mg | twice daily", "Meets all 3 dose-reduction criteria."),
    ("P028", "dose", 79, "M", 70, 2.0, "No", "atrial fibrillation", "none",
     "dabigatran | 150 mg | twice daily", ""),
    ("P029", "dose", 84, "F", 52, 1.5, "No", "after hip surgery (DVT prophylaxis)", "none",
     "enoxaparin | 40 mg SC | once daily", ""),
    ("P030", "dose", 77, "M", 72, 1.7, "No", "atrial fibrillation", "none",
     "rivaroxaban | 20 mg | once daily", ""),
    ("P031", "dose", 52, "M", 85, 0.7, "No", "atrial fibrillation", "none",
     "edoxaban | 60 mg | once daily", "Kidney function too HIGH for edoxaban in AF."),
    ("P032", "dose", 74, "F", 62, 1.3, "No", "atrial fibrillation", "none",
     "dofetilide | 500 mcg | twice daily", ""),
    # --- Several problems at once ------------------------------------------
    ("P033", "multiple", 81, "F", 54, 0.9, "No", "atrial fibrillation; osteoarthritis; hyperlipidaemia", "none",
     "apixaban | 5 mg | twice daily; amiodarone | 200 mg | once daily; simvastatin | 40 mg | once daily; "
     "ibuprofen | 400 mg | three times daily; metoprolol succinate | 50 mg | once daily",
     "Acceptable extra (not scored): NSAID caution in an elderly patient with reduced kidney function."),
    ("P034", "multiple", 70, "M", 88, 1.0, "No", "atrial fibrillation; heart failure", "none",
     "warfarin | 5 mg | once daily; amiodarone | 200 mg | once daily (new); digoxin | 0.125 mg | once daily", ""),
    # --- Safe cases: NO warning expected -----------------------------------
    ("P035", "negative", 58, "M", 85, 1.0, "No", "hypertension; hyperlipidaemia", "none",
     "metoprolol succinate | 50 mg | once daily; atorvastatin | 40 mg | once daily; losartan | 50 mg | once daily", ""),
    ("P036", "negative", 63, "F", 70, 0.8, "No", "atrial fibrillation", "none",
     "apixaban | 5 mg | twice daily; metoprolol succinate | 50 mg | once daily; atorvastatin | 20 mg | once daily",
     "Correct apixaban dose (no reduction criteria met)."),
    ("P037", "negative", 55, "M", 90, 0.9, "No", "type 2 diabetes; hypertension", "none",
     "metformin | 1000 mg | twice daily; valsartan | 160 mg | once daily; rosuvastatin | 10 mg | once daily", "eGFR is normal."),
    ("P038", "negative", 67, "M", 80, 1.0, "No", "heart failure", "none",
     "carvedilol | 12.5 mg | twice daily; furosemide | 40 mg | once daily; rosuvastatin | 20 mg | once daily", ""),
    ("P039", "negative", 72, "F", 65, 0.9, "No", "atrial fibrillation", "none",
     "rivaroxaban | 20 mg | once daily; metoprolol succinate | 50 mg | once daily; atorvastatin | 20 mg | once daily",
     "CrCl is above 50, so 20 mg is correct."),
    ("P040", "negative", 60, "M", 82, 1.0, "No", "coronary stent placed 1 month ago", "none",
     "clopidogrel | 75 mg | once daily; aspirin | 81 mg | once daily; atorvastatin | 80 mg | once daily; metoprolol succinate | 50 mg | once daily",
     "Trap: intended dual antiplatelet therapy. A bleeding-risk note is acceptable but must not be a high-severity error."),
    ("P041", "negative", 30, "F", 65, 0.6, "Yes (24 weeks)", "hypertension in pregnancy", "none",
     "hydralazine | 25 mg | four times daily", "Trap: pregnant, but hydralazine is not contraindicated."),
    ("P042", "negative", 80, "M", 75, 1.2, "No", "atrial fibrillation", "none",
     "apixaban | 5 mg | twice daily; metoprolol succinate | 50 mg | once daily",
     "Trap: only 1 of 3 apixaban criteria (age 80), so 5 mg is correct."),
    ("P043", "negative", 70, "M", 80, 1.0, "No", "hypertension; hyperlipidaemia", "none",
     "simvastatin | 10 mg | once daily; diltiazem ER | 180 mg | once daily",
     "Trap: simvastatin 10 mg is within the limit for diltiazem."),
    ("P044", "negative", 50, "F", 70, 0.8, "No", "hypertension", "none",
     "metoprolol succinate | 100 mg | once daily; hydrochlorothiazide | 25 mg | once daily", ""),
]

# ---------------------------------------------------------------------------
# Expected flags: (patient, type, drugs, severity, what the system should say,
#                  source drug label, source section field, evidence phrase)
# The evidence phrase is copied from the label; verify_test_set.py checks it.
# ---------------------------------------------------------------------------
F = [
    ("P001", "interaction", "warfarin + aspirin", "major", "Increased bleeding risk; check indication for aspirin",
     "warfarin", "drug_interactions", "Antiplatelet Agents aspirin"),
    ("P002", "interaction", "warfarin + fluconazole", "major", "Fluconazole (CYP2C9 inhibitor) raises INR; monitor INR closely / reduce warfarin",
     "warfarin", "drug_interactions", "cotrimoxazole, etravirine, fluconazole"),
    ("P003", "interaction", "warfarin + sulfamethoxazole/trimethoprim", "major", "Co-trimoxazole raises INR; monitor INR closely",
     "warfarin", "drug_interactions", "capecitabine, cotrimoxazole"),
    ("P004", "interaction", "simvastatin + clarithromycin", "contraindicated", "Strong CYP3A4 inhibitor: contraindicated (myopathy/rhabdomyolysis)",
     "simvastatin", "contraindications", "Concomitant use of strong CYP3A4 inhibitors (select azole anti-fungals, macrolide antibiotics"),
    ("P005", "interaction", "simvastatin + amiodarone", "major", "Simvastatin must not exceed 20 mg daily with amiodarone",
     "simvastatin", "dosage_and_administration", "Patients taking Amiodarone, Amlodipine, or Ranolazine Do not exceed simvastatin tablets USP 20 mg once daily"),
    ("P006", "interaction", "simvastatin + diltiazem", "major", "Simvastatin must not exceed 10 mg daily with diltiazem",
     "simvastatin", "dosage_and_administration", "Patients taking Verapamil, Diltiazem, or Dronedarone Do not exceed simvastatin tablets USP 10 mg once daily"),
    ("P007", "interaction", "dofetilide + verapamil", "contraindicated", "Concomitant verapamil is contraindicated (QT prolongation/torsade risk)",
     "dofetilide", "contraindications", "The concomitant use of verapamil"),
    ("P008", "interaction", "dofetilide + hydrochlorothiazide", "contraindicated", "Concomitant hydrochlorothiazide is contraindicated",
     "dofetilide", "contraindications", "The concomitant use of hydrochlorothiazide (alone or in combinations such as with triamterene) with dofetilide is contraindicated"),
    ("P009", "interaction", "clopidogrel + omeprazole", "major", "Avoid omeprazole: reduces clopidogrel's antiplatelet effect",
     "clopidogrel", "drug_interactions", "Avoid concomitant use of clopidogrel with omeprazole or esomeprazole"),
    ("P010", "interaction", "apixaban + itraconazole", "major", "Combined P-gp and strong CYP3A4 inhibitor: reduce apixaban dose by 50% or avoid",
     "apixaban", "dosage_and_administration", "strong cytochrome P450 3A4 (CYP3A4) inhibitors (e.g., ketoconazole, itraconazole, ritonavir)"),
    ("P011", "interaction", "rivaroxaban + rifampin", "major", "Combined P-gp and strong CYP3A inducer: avoid (reduced anticoagulation)",
     "rivaroxaban", "drug_interactions", "(e.g., carbamazepine, phenytoin, rifampin, St. John's wort)"),
    ("P012", "interaction", "ticagrelor + aspirin 325 mg", "major", "Aspirin maintenance dose must be 75-100 mg with ticagrelor",
     "ticagrelor", "dosage_and_administration", "daily maintenance dose of aspirin of 75 to 100 mg"),
    ("P013", "interaction", "digoxin + amiodarone", "major", "Amiodarone raises digoxin ~70%: measure levels, reduce digoxin dose 30-50%",
     "digoxin", "drug_interactions", "decreasing dose by approximately 30 to 50%"),
    ("P014", "interaction", "spironolactone + potassium chloride", "major", "Hyperkalaemia risk: potassium supplements should not ordinarily be given",
     "spironolactone", "warnings", "Potassium supplementation, either in the form of medication"),
    ("P015", "interaction", "sildenafil + isosorbide mononitrate", "contraindicated", "Nitrates contraindicated with sildenafil (severe hypotension)",
     "sildenafil", "contraindications", "organic nitrates or organic nitrites in any form"),
    ("P016", "interaction", "ranolazine + clarithromycin", "contraindicated", "Strong CYP3A inhibitor: contraindicated",
     "ranolazine", "contraindications", "Strong CYP3A inhibitors (e.g., ketoconazole, clarithromycin"),
    ("P017", "interaction", "dronedarone + clarithromycin", "contraindicated", "Potent CYP3A inhibitor: contraindicated",
     "dronedarone", "drug_interactions", "itraconazole, voriconazole, ritonavir, clarithromycin, and nefazodone is contraindicated"),
    ("P018", "interaction", "warfarin + naproxen", "major", "NSAID increases bleeding risk",
     "warfarin", "drug_interactions", "mefenamic acid, naproxen"),
    ("P019", "interaction", "apixaban + carbamazepine", "major", "Combined P-gp and strong CYP3A4 inducer: avoid (reduced apixaban levels)",
     "apixaban", "drug_interactions", "combined P-gp and strong CYP3A4 inducers reduces blood levels of apixaban"),
    ("P020", "interaction", "enalapril + potassium chloride", "moderate", "Hyperkalaemia risk: monitor serum potassium",
     "enalapril", "drug_interactions", "Agents Increasing Serum Potassium"),
    ("P021", "contraindication", "warfarin (pregnancy)", "contraindicated", "Warfarin contraindicated in pregnancy (no mechanical valve)",
     "warfarin", "contraindications", "contraindicated in women who are pregnant except in pregnant women with mechanical heart valves"),
    ("P022", "contraindication", "losartan (pregnancy)", "contraindicated", "Fetal toxicity: discontinue losartan when pregnancy is detected",
     "losartan", "boxed_warning", "When pregnancy is detected, discontinue losartan potassium as soon as possible"),
    ("P023", "contraindication", "metformin (eGFR < 30)", "contraindicated", "Metformin contraindicated with eGFR below 30",
     "metformin", "contraindications", "Severe renal impairment (eGFR below 30"),
    ("P024", "contraindication", "prasugrel (prior TIA)", "contraindicated", "Prasugrel contraindicated after prior TIA or stroke",
     "prasugrel", "contraindications", "Prior transient ischemic attack or stroke"),
    ("P025", "contraindication", "propranolol (asthma)", "contraindicated", "Propranolol contraindicated in bronchial asthma",
     "propranolol", "contraindications", "bronchial asthma"),
    ("P026", "contraindication", "sulfamethoxazole/trimethoprim (sulfonamide allergy)", "contraindicated", "Contraindicated with known sulfonamide hypersensitivity",
     "sulfamethoxazole and trimethoprim", "contraindications", "known hypersensitivity to trimethoprim or sulfonamides"),
    ("P027", "dose", "apixaban", "major", "Age >=80, weight <=60 kg, SCr >=1.5: dose should be 2.5 mg twice daily",
     "apixaban", "dosage_and_administration", "at least 2 of the following characteristics: age greater than or equal to 80 years"),
    ("P028", "dose", "dabigatran", "major", "CrCl 15-30 mL/min: dose should be 75 mg twice daily",
     "dabigatran", "dosage_and_administration", "For patients with CrCl 15 to 30 mL/min: 75 mg orally, twice daily"),
    ("P029", "dose", "enoxaparin", "major", "CrCl <30 mL/min: prophylaxis dose should be 30 mg once daily",
     "enoxaparin", "dosage_and_administration", "severe renal impairment (creatinine clearance"),
    ("P030", "dose", "rivaroxaban", "major", "AF with CrCl <=50 mL/min: dose should be 15 mg once daily",
     "rivaroxaban", "dosage_and_administration", "CrCl ≤50 mL/min"),
    ("P031", "dose", "edoxaban", "major", "AF with CrCl >95 mL/min: do not use edoxaban (reduced efficacy)",
     "edoxaban", "dosage_and_administration", "Do not use SAVAYSA in patients with CrCL > 95 mL/min"),
    ("P032", "dose", "dofetilide", "major", "CrCl 20-<40 mL/min: dose should be 125 mcg twice daily",
     "dofetilide", "dosage_and_administration", "125 mcg twice dail"),
    ("P033", "dose", "apixaban", "major", "Age >=80 and weight <=60 kg (2 of 3 criteria): dose should be 2.5 mg twice daily",
     "apixaban", "dosage_and_administration", "at least 2 of the following characteristics: age greater than or equal to 80 years"),
    ("P033", "interaction", "simvastatin + amiodarone", "major", "Simvastatin must not exceed 20 mg daily with amiodarone",
     "simvastatin", "dosage_and_administration", "Patients taking Amiodarone, Amlodipine, or Ranolazine Do not exceed simvastatin tablets USP 20 mg once daily"),
    ("P033", "interaction", "apixaban + ibuprofen", "major", "NSAID increases bleeding risk with apixaban",
     "apixaban", "drug_interactions", "chronic NSAID use increases the risk of bleeding"),
    ("P034", "interaction", "warfarin + amiodarone", "major", "Amiodarone (CYP2C9 inhibitor) raises INR: monitor INR, reduce warfarin",
     "warfarin", "drug_interactions", "CYP2C9 amiodarone"),
    ("P034", "interaction", "digoxin + amiodarone", "major", "Amiodarone raises digoxin: measure levels, reduce digoxin dose 30-50%",
     "digoxin", "drug_interactions", "decreasing dose by approximately 30 to 50%"),
]


def main():
    # Split: alternate within each category so dev and test both cover every type.
    split, seen = {}, {}
    for p in P:
        n = seen.get(p[1], 0)
        split[p[0]] = "dev" if n % 2 == 0 else "test"
        seen[p[1]] = n + 1

    n_flags = {}
    for f in F:
        n_flags[f[0]] = n_flags.get(f[0], 0) + 1

    with open("patients.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "split", "category", "age", "sex", "weight_kg", "serum_creatinine_mg_dl",
                    "crcl_ml_min", "egfr_ml_min_1_73m2", "pregnant", "conditions", "allergies",
                    "medications", "n_expected_flags", "notes", "review_status"])
        for (pid, cat, age, sex, wt, scr, preg, cond, allg, meds, notes) in P:
            w.writerow([pid, split[pid], cat, age, sex, wt, scr,
                        round(crcl_cockcroft_gault(age, wt, scr, sex), 1),
                        round(egfr_ckd_epi_2021(age, scr, sex), 1),
                        preg, cond, allg, meds, n_flags.get(pid, 0), notes, "needs pharmacist review"])

    with open("expected_flags.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["flag_id", "patient_id", "split", "flag_type", "drugs", "severity", "expected_message",
                    "source_drug", "source_section", "evidence_phrase", "review_status"])
        for i, (pid, typ, drugs, sev, msg, sdrug, sec, ev) in enumerate(F, 1):
            w.writerow([f"F{i:03d}", pid, split[pid], typ, drugs, sev, msg, sdrug, sec, ev, "needs pharmacist review"])
    print(f"{len(P)} patients, {len(F)} expected flags")


if __name__ == "__main__":
    main()
