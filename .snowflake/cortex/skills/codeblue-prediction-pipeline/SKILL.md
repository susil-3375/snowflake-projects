---
name: codeblue-prediction-pipeline
description: >
  End-to-end Code Blue (cardiac arrest) prediction ML pipeline on Snowflake: synthetic patient,
  flowsheet and lab data, Snowpark feature engineering, Feature Store, multi-model training with
  experiment tracking, Model Registry with SHAP explainability, batch scoring, final results table.
  Triggers: code blue, codeblue model, patient deterioration, critical care prediction, ICU risk.
---

# Code Blue Prediction Pipeline

First load the `machine-learning` skill (and `notebooks-in-workspaces`), then follow the instructions below.

Example prompts:
- "Run the code blue prediction pipeline"
- "Build the CodeBlue model in MY_DB.MY_SCHEMA using MY_WH"

Defaults: database DEMO_DB, schema CLINIC, Feature Store schema FEATURES, warehouse DEMO_WH —
unless the user specifies others. Substitute user-provided names everywhere below.

TIME BUDGET: ~10 minutes. Favor speed. Use defaults; no clarifying questions.
COMPUTE: Warehouse DEMO_WH (XSMALL, AUTO_SUSPEND=60) for SQL. For the notebook kernel, reuse an
existing notebook service, or create the default CPU notebook service if none exists (it uses the
system pool). Do NOT create custom compute pools, model inference services, or ML Jobs.

## Autonomy
- Invoking this skill authorizes all DDL/DML, notebook edits and cell runs inside the target
  database, plus creating the target warehouse.
- No plan mode, no ask_user_question, no check-ins between phases. Finish phase 6 before replying.
- Never drop/modify objects outside the target database. Inside it, CREATE OR REPLACE is fine.
- On failure: fix and retry once; if it still fails, note it in that phase's markdown, simplify, continue.
- Snowsight consent cards for write SQL / bash are UI-enforced; the user approves them — just proceed.

## Targets
- Schema CLINIC (tables, experiment, model); schema FEATURES (Feature Store).
- Always use fully qualified names (DEMO_DB.CLINIC.<obj>) in Snowpark/SQL — the Feature Store
  switches the session schema, so unqualified names break feature-view creation.
- Model CODEBLUE_MODEL, version V1_0 (registry versions must be identifiers; "v1.0" is not valid).
- Experiment CODE_BLUE_EXPERIMENT. Seed 42 everywhere.

## Notebook: notebooks/CodeBlue_Prediction.ipynb (create; if it exists, replace its cells)
- Cell 1 (markdown): title "Code Blue Prediction Model & Critical Care Optimization Platform",
  one-paragraph overview, numbered TOC of the 6 phases, and a one-line assumptions note.
- Cell 2 (python): setup only — imports, get_active_session(), USE WAREHOUSE/DATABASE/SCHEMA,
  constants DB, SCHEMA, WH, FEATURES, EXPERIMENT, MODEL_NAME, MODEL_VERSION, FQ_MODEL, FV_NAME,
  FV_VERSION, SEED. All later cells reuse these.
- Each phase = one markdown cell (what/why/assumptions) + one code cell. Phase 6 = SQL cells.

## Phase 1 — Training dataset (PATIENT_CODEBLUE)
- Generate 1,000 synthetic encounters, ENTRY_DATE within the last 90 days (hourly granularity).
- Source tables SRC_PATIENT, SRC_FLOWSHEET, SRC_VITAL_LAB (write_pandas, use_logical_type=True),
  joined on PRIMARYMRN (format MRN1000000…) into DEMO_DB.CLINIC.PATIENT_CODEBLUE.
- Columns:
  - Patient: PRIMARYMRN, BIRTHDATE (DATE), GENDER (Male/Female), AGE, HEIGHT_CMS, WEIGHT_LBS,
    ENTRY_DATE (TIMESTAMP_NTZ), DISCHARGE_DATE (TIMESTAMP_NTZ)
  - Flowsheet: RESPIRATION, SYSTOLIC_BP, DIASTOLIC_BP, TEMPERATURE, PULSE_OXIMETRY, O2_FLOW_RATE, PULSE
  - Lab: ABS_LYMPHOCYTE, ABS_NEUTROPHIL, ALBUMIN, BLOOD_CULTURE (0/1), BNP, BUN, CALCIUM
  - Target: CODEBLUE (0/1)
- Realism: a latent acuity score (+ age effect) drives correlated vitals/labs (high acuity → higher
  RR, pulse, BNP, BUN, neutrophils, temp, O2 flow; lower SBP, SpO2, albumin, calcium, lymphocytes);
  clip to physiologic ranges. CODEBLUE drawn from a logistic model on these signals, target ~8–10%
  prevalence. ~3% missing values injected into vitals/labs/height/weight.
- DISCHARGE_DATE always populated; LOS (gamma) grows with acuity and Code Blue (recent admissions
  may have a projected future discharge — note this).
- Print row count and class balance.

## Phase 2 — Feature engineering + Feature Store
- Snowpark pipeline (pushed down to SQL so training and inference share identical logic):
  median imputation (medians computed once, baked in as literals); GENDER via sklearn LabelEncoder
  mapping expressed as a CASE; derived BMI, MEAN_ARTERIAL_PRESSURE, SHOCK_INDEX,
  NEUTROPHIL_LYMPH_RATIO, HYPOXIA_FLAG (SpO2<92); date features ENTRY_DOW, ENTRY_MONTH.
  Cast all features to DOUBLE.
- LEAKAGE RULE: compute LOS_DAYS and ADMISSION_DURATION_HRS for reporting only — exclude them
  from model features (only known at discharge and correlated with the label by construction).
- Scaling: none for tree models; StandardScaler inside the LogisticRegression sklearn Pipeline.
- FeatureStore(database=DB, name=FEATURES, default_warehouse=WH, CREATE_IF_NOT_EXIST);
  Entity PATIENT (join key PRIMARYMRN; tolerate "already exists");
  FeatureView FV_CODE_BLUE_PREDICTION, version V1 (physical FV_CODE_BLUE_PREDICTION$V1),
  timestamp_col ENCOUNTER_TS (= ENTRY_DATE), refresh_freq=None, overwrite=True.
- attach_feature_desc text must contain NO quotes, braces or slashes (it becomes SQL comments),
  e.g. "LabelEncoder Female=0, Male=1".
- Spine (PRIMARYMRN, ENCOUNTER_TS, CODEBLUE) → fs.generate_training_set(spine_df=…,
  features=[fv], spine_timestamp_col='ENCOUNTER_TS', spine_label_cols=['CODEBLUE']).
- Print training-set shape, positives, FEATURE_COLS.

## Phase 3 — Model training & evaluation (notebook kernel)
- 80/20 stratified split, random_state=42.
- Families, 2 configs each (light grid): XGBoost (scale_pos_weight), LightGBM (class_weight
  balanced, subsample_freq=1), RandomForest (balanced_subsample), LogisticRegression
  (StandardScaler pipeline, balanced), sklearn GradientBoostingClassifier (balanced sample_weight).
- Tune n_estimators, learning_rate, max_depth, subsample, min_samples_split where applicable.
- Metrics at threshold 0.5: accuracy, precision, recall, F1, ROC-AUC, confusion matrix (tn/fp/fn/tp).
- ExperimentTracking(session, database_name=DB, schema_name=SCHEMA).set_experiment(CODE_BLUE_EXPERIMENT);
  one run per config named <FAMILY>_<i>_<timestamp>; log_params + log_metrics per run.
- Print a leaderboard; champion = highest test ROC-AUC.

## Phase 4 — Model Registry
- Registry(session, database_name=DB, schema_name=SCHEMA); delete existing CODEBLUE_MODEL first (idempotent).
- log_model(champion, model_name=CODEBLUE_MODEL, version_name=V1_0, sample_input_data=X_train
  head(100) as float64, target_platforms=['WAREHOUSE'], options={'enable_explainability': True},
  metrics={performance metrics, confusion_matrix, model_params, algorithm, experiment, run,
  training_metadata, feature_definitions (FV FQN, features, medians, gender encoding),
  feature_importance}, comment=...). If explainability fails, retry without it.
- Set V1_0 as the default (= approved) version. Print show_functions() and top 8 features.

## Phase 5 — Production inference
- Load default version of CODEBLUE_MODEL from the registry.
- fs.retrieve_feature_values(spine of all 1,000 PATIENT_CODEBLUE rows, [fv]) → same FEATURE_COLS.
- mv.run(function_name='predict_proba') → probability = last output column; prediction = p ≥ 0.5.
- mv.run(function_name='explain') → per-row SHAP; build JSON {method, top_drivers: top 5
  {feature, value, contribution, direction}}. Fallback: importance × z-score.
- Write DEMO_DB.CLINIC.CODEBLUE_SCORING_RESULTS: PRIMARYMRN, CODEBLUE_PREDICTION,
  CODEBLUE_PROBABILITY, MODEL_VERSION ('CODEBLUE_MODEL/V1_0'), SCORING_TIMESTAMP,
  EXPLAINABILITY_ATTRIBUTES (VARIANT via PARSE_JSON). Drop the staging table afterwards.

## Phase 6 — Final dataset (SQL cells)
- CREATE OR REPLACE DEMO_DB.CLINIC.CODEBLUE_PREDICTION_RESULTS AS p.* from PATIENT_CODEBLUE LEFT
  JOIN CODEBLUE_SCORING_RESULTS on PRIMARYMRN, appending CODEBLUE_PREDICTION,
  CODEBLUE_PROBABILITY, MODEL_VERSION, SCORING_TIMESTAMP, EXPLAINABILITY_ATTRIBUTES.
- Validation query: total rows (=1000), scored, actual vs predicted Code Blue, true positives,
  avg LOS overall vs Code Blue, high-risk count (p ≥ 0.7), model version, and the most common
  top SHAP driver.

## Final response (only after phase 6)
1. Table of all created objects (FQN, type, notes).
2. Business recap: champion and test metrics, flagged/high-risk counts, top risk drivers,
   LOS-by-criticality, and caveats (synthetic data; full-dataset scoring includes training rows).
