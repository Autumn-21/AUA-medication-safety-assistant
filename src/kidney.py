"""Kidney function calculations. Done in code, never by the AI (AI models make arithmetic mistakes)."""


def crcl_cockcroft_gault(age: float, weight_kg: float, scr_mg_dl: float, sex: str) -> float:
    """Creatinine clearance in mL/min (Cockcroft-Gault, actual body weight).

    Most FDA labels base renal dose changes on this value.
    """
    if scr_mg_dl <= 0:
        raise ValueError("serum creatinine must be positive")
    value = (140 - age) * weight_kg / (72 * scr_mg_dl)
    return value * 0.85 if sex.upper().startswith("F") else value


def egfr_ckd_epi_2021(age: float, scr_mg_dl: float, sex: str) -> float:
    """eGFR in mL/min/1.73 m2 (CKD-EPI 2021, race-free). Used e.g. by the metformin label."""
    female = sex.upper().startswith("F")
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    value = (142 * min(scr_mg_dl / k, 1) ** a * max(scr_mg_dl / k, 1) ** -1.200 * 0.9938 ** age)
    return value * 1.012 if female else value
