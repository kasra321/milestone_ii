# MIMIC-IV-ED Data Description

## Abstract

MIMIC-IV-ED is a comprehensive, freely available database of emergency department (ED) admissions at the Beth Israel Deaconess Medical Center from 2011 to 2019. The database contains approximately 425,000 ED stays, including vital signs, triage information, medication reconciliation, medication administration, and discharge diagnoses. All data comply with HIPAA Safe Harbor provisions, facilitating diverse education and research studies.

---

## Background

The ED is a high-demand, resource-limited environment where patients are triaged for further care. MIMIC-IV-ED supports data-driven analyses by providing a large, deidentified dataset of ED admissions linked to MIMIC-IV and MIMIC-CXR databases.

---

## Methods

Data were extracted from XML files and transformed into a denormalized relational database for simplified analysis. Key details:

- **Deidentification**: Patient identifiers replaced with randomized surrogates; temporal consistency maintained within patient data.
- **Schema**: Six tables—`edstays`, `diagnosis`, `medrecon`, `pyxis`, `triage`, and `vitalsign`.
- **Data Cleaning**: Observations deduplicated; erroneous or implausible values removed.

---

## Data Description

### edstays

Tracks patient stays in the ED. Key columns:

- **subject_id**: Unique patient identifier.
- **stay_id**: Unique ED stay identifier.
- **intime/outtime**: Admission/discharge times.
- **gender/race**: Patient demographics.
- **arrival_transport**: Admission mechanism (e.g., AMBULANCE, WALK IN).
- **disposition**: Discharge location (e.g., HOME, ADMITTED).

### diagnosis

Provides ICD-9 or ICD-10 coded diagnoses. Columns:

- **icd_code**: Diagnosis code.
- **icd_title**: Textual description.
- **seq_num**: Relevance order (1 = highest).

### medrecon

Details medication reconciliation. Columns:

- **name**: Medicine name.
- **gsn/ndc**: Generic/National Drug Codes.
- **etccode/etcdescription**: Drug ontology group.

### pyxis

Lists medications dispensed via the BD Pyxis MedStation. Columns:

- **charttime**: Dispensation time.
- **name**: Medication description.
- **gsn**: Generic Sequence Number.

### triage

Captures patient data at triage. Columns:

- **temperature, heartrate, resprate, o2sat, sbp, dbp**: Vital signs.
- **acuity**: Severity level (1 = highest).
- **chiefcomplaint**: Patient-reported reason for visit.

### vitalsign

Logs periodic vital signs during the stay. Columns:

- **charttime**: Time recorded.
- **rhythm**: Heart rhythm.

---

## Usage Notes

### Organization

MIMIC-IV-ED uses a star schema, with `edstays` as the central table linking all other tables via `stay_id`.

### Data Linkage

MIMIC-IV-ED links with:

- **MIMIC-IV**: For additional hospital stay details (e.g., lab results, medications).
- **MIMIC-CXR**: For chest x-rays and reports.

### Limitations

Data reflects routine clinical care, with potential biases, missing documentation, and implausible values. Researchers must address these limitations.

---

## Release Notes

### MIMIC-IV-ED v2.2 (January 5, 2023)

- Removed test set patients.
- Updated `edstays` table: Removed 22,625 `stay_id`.

### MIMIC-IV-ED v2.0 (May 2022)

- Schema changes: Added `gender`, `race`, `arrival_transport`, and `disposition` to `edstays`.
- Corrected `outtime` column issues.
- Updated `pain` column in `triage` to include free-text.

### MIMIC-IV-ED v1.0 (June 3, 2021)

- Initial release with six tables: `edstays`, `diagnosis`, `medrecon`, `pyxis`, `triage`, and `vitalsign`.
