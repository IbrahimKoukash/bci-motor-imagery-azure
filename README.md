# CCAMIC Neuro-Decode · BCI Competition IV 2a
An end-to-end motor-imagery EEG analysis pipeline, from raw recording to
cross-subject decoding, delivered as a live web application on Microsoft Azure,
with results published to Microsoft Fabric and Power BI.

Built at the System Engineering Laboratory, Department of Mechanical Engineering,
Kyushu University.

**Live demo:** https://ibrahim-bci-pipeline.azurewebsites.net

---

## What it does
The [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/) processes 22-channel EEG from nine people while they imagine one of four movements (left hand, right hand, feet, or tongue). This project takes a raw session and:

1. **Cleans it** — FIR band-pass, average reference, and ICA-based removal of
   eye-movement artefacts.
2. **Epochs it** — cue-locked trials, baseline-corrected, cropped to the
   event-related desynchronisation (ERD) window where the motor signal is
   strongest.
3. **Decodes it** — eight feature extractors × five classifiers, scored with
   leak-safe cross-validation (GroupKFold on trial id) against the 25% chance
   level for four classes.
4. **Compares across subjects** — a bounded sweep over all nine participants,
   with results persisted for dashboarding.

Everything runs behind a four-tab web UI; no local install is needed to try it.
