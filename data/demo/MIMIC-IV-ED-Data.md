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

Patient stays are tracked in the `edstays` table. Each row has a unique `stay_id`, representing a unique patient stay in the ED. Key columns:

- **subject_id**: Unique patient identifier (links across multiple stays)
- **hadm_id**: Hospital admission ID (if admitted)
- **stay_id**: Unique ED stay identifier
- **intime/outtime**: Admission/discharge times
- **gender/race**: Patient demographics
- **arrival_transport**: Admission mechanism (AMBULANCE, HELICOPTER, WALK IN, UNKNOWN, OTHER)
- **disposition**: Discharge location (ADMITTED, ELOPED, EXPIRED, HOME, LEFT AGAINST MEDICAL ADVICE, LEFT WITHOUT BEING SEEN, TRANSFER, OTHER)

Note: `subject_id` can link MIMIC-IV-ED with MIMIC-IV for additional information (e.g., age) and with MIMIC-CXR's PatientID DICOM attribute for chest x-rays.

### diagnosis

Provides coded diagnoses using ICD-9 or ICD-10. Columns:

- **subject_id**: Patient identifier
- **stay_id**: ED stay identifier
- **seq_num**: Pseudo-order (1-9, 1 = highest relevance)
- **icd_code**: Coded diagnosis
- **icd_version**: ICD version (9 or 10)
- **icd_title**: Textual description

Note: Contains only ED-specific diagnoses. Hospital admission diagnoses are separate and available in MIMIC-IV.

### medrecon

Medicine reconciliation table listing medications taken prior to ED stay. Columns:

- **subject_id/stay_id**: Patient and stay identifiers
- **charttime**: Documentation time
- **name**: Medicine description
- **gsn**: Generic Sequence Number (0 if missing)
- **ndc**: National Drug Code (0 if missing)
- **etc_rn**: Sequential identifier for multiple classifications
- **etccode/etcdescription**: Drug ontology group code and description

Multiple rows may exist per medication due to multiple classification groups.

### pyxis

Records medications dispensed via BD Pyxis MedStation. Columns:

- **subject_id/stay_id**: Patient and stay identifiers
- **charttime**: Dispensation time
- **med_rn**: Multiple medication delineator
- **name**: Medication description and formulation
- **gsn_rn**: GSN delineator
- **gsn**: Generic Sequence Number (0 if missing)

Note: Not all medications (e.g., large fluid volumes) are recorded in this table.

### triage

Contains initial patient assessment data. Columns:

- **subject_id/stay_id**: Patient and stay identifiers
- **Vital Signs**:
  - temperature
  - heartrate
  - resprate
  - o2sat
  - sbp (systolic blood pressure)
  - dbp (diastolic blood pressure)
- **pain**: Patient-reported pain level
- **acuity**: Severity level (1-5, 1 = highest)
- **chiefcomplaint**: Patient's reason for visit (comma-separated, PHI replaced with "___")

### vitalsign

Records aperiodic vital signs during stay. Columns:

- **subject_id/stay_id**: Patient and stay identifiers
- **charttime**: Recording time
- **Vital Signs**:
  - temperature
  - heartrate
  - resprate
  - o2sat
  - sbp
  - dbp
- **rhythm**: Heart rhythm
- **pain**: Pain level

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
