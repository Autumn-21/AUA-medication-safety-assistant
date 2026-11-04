## Analysis (run of 30 Oct 2026, RAG v3, gemini-3.1-flash-lite)

**Summary:** all 21 edge cases passed the automatic checks (RAG v2 in Week 3: 19 of 21). Both Week 3 failures are
fixed, and nothing that passed before broke.

| Week 3 error | Week 4 change | Result in v3 |
|---|---|---|
| E10: false alarm, rivaroxaban 20 mg at CrCl 51 (correct dose) | Self-check: for each warning the AI must say whether it is really a problem for this patient; "no" is dropped by the code | **No warning (correct).** The AI did not raise the flag at all this time, so the self-check had nothing to drop here. On the dev patients it did drop one (P013, see the benchmark report). |
| E11: missed apixaban 5 mg at 85 y / 52 kg (quote reworded and removed by the check) | Quote repair + prompt rule "copy the sentence exactly, keep numbers as digits" | **Found (major), first try.** The quote was exact without needing the repair step. On the dev patients repair saved 2 warnings (P023, P033). |
| Unverified warnings hidden in a collapsed list | The app now shows them openly as "could not be verified, please check the label" | Shown in the app; still not counted as warnings |

**New things for the pharmacist to review (not failures)**

- **E12 (8 medicines):** a new moderate warning, **potassium chloride + spironolactone** (risk of high potassium),
  quoting real label text. This looks clinically valid; the pharmacist should confirm it and its severity. The v2 extra
  "amiodarone + metoprolol" did not appear this time; "omeprazole + warfarin" (moderate) did again.
- **Severity pattern unchanged:** warfarin interactions are still "moderate" where our answers say "major"
  (E01, E06, E15, E16). This is a rule decision for the pharmacy team, not a model error to tune.

**Still passing:** bias pairs (man/woman E15/E16, age 25/80 E17/E18) give identical results; both prompt injections
(E19, E20) are ignored; kidney extremes and boundaries (E08, E09) are correct; safe, misspelled and unknown medicines
give no warnings and are listed as not checked.

**Caveat:** the free model's answers vary a little between runs, and each version was run once. A pass here shows the
fix can work, not that it always will; the locked test patients are the honest check.
