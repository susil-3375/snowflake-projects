CREATE OR REPLACE SEMANTIC VIEW DEMO_DB.CLINIC.CODEBLUE_PREDICTION_SV
  TABLES (
    predictions AS DEMO_DB.CLINIC.CODEBLUE_PREDICTION_RESULTS
      PRIMARY KEY (PRIMARYMRN)
      WITH SYNONYMS = ('patients', 'code blue predictions', 'patient risk scores')
      COMMENT = 'One row per patient encounter scored by the Code Blue ML model, with demographics, admission/discharge dates, latest flowsheet vitals, lab results, actual outcome, predicted outcome, probability and SHAP explainability.'
  )
  FACTS (
    -- Demographics
    predictions.age_years AS AGE COMMENT = 'Patient age in years at time of admission (range 18-98).',
    predictions.height_cms AS HEIGHT_CMS COMMENT = 'Patient height in centimeters.',
    predictions.weight_lbs AS WEIGHT_LBS COMMENT = 'Patient body weight in pounds.',
    predictions.length_of_stay_days AS ROUND(DATEDIFF('hour', ENTRY_DATE, DISCHARGE_DATE) / 24, 1)
      WITH SYNONYMS = ('LOS', 'days in hospital') COMMENT = 'Hospital length of stay in days, computed from hours between admission and discharge divided by 24, rounded to 1 decimal.',
    -- Flowsheet vitals
    predictions.respiration AS RESPIRATION WITH SYNONYMS = ('respiratory rate', 'RR', 'breaths per minute')
      COMMENT = 'Flowsheet respiratory rate in breaths per minute. Normal adult range 12-20.',
    predictions.systolic_bp AS SYSTOLIC_BP WITH SYNONYMS = ('SBP', 'systolic blood pressure')
      COMMENT = 'Flowsheet systolic blood pressure in mmHg. Below 90 indicates hypotension.',
    predictions.diastolic_bp AS DIASTOLIC_BP WITH SYNONYMS = ('DBP', 'diastolic blood pressure')
      COMMENT = 'Flowsheet diastolic blood pressure in mmHg.',
    predictions.temperature AS TEMPERATURE WITH SYNONYMS = ('body temperature', 'temp')
      COMMENT = 'Flowsheet body temperature reading.',
    predictions.pulse_oximetry AS PULSE_OXIMETRY WITH SYNONYMS = ('SpO2', 'oxygen saturation', 'O2 sat')
      COMMENT = 'Flowsheet peripheral oxygen saturation percentage. Below 92 suggests hypoxemia.',
    predictions.o2_flow_rate AS O2_FLOW_RATE WITH SYNONYMS = ('oxygen flow', 'supplemental oxygen')
      COMMENT = 'Supplemental oxygen flow rate in liters per minute; 0 means room air.',
    predictions.pulse AS PULSE WITH SYNONYMS = ('heart rate', 'HR')
      COMMENT = 'Flowsheet heart rate in beats per minute. Above 100 indicates tachycardia.',
    predictions.shock_index AS PULSE / NULLIF(SYSTOLIC_BP, 0)
      COMMENT = 'Shock index = heart rate / systolic BP. Values above 0.9 indicate elevated risk of hemodynamic instability.',
    -- Labs
    predictions.abs_lymphocyte AS ABS_LYMPHOCYTE WITH SYNONYMS = ('lymphocyte count', 'ALC')
      COMMENT = 'Absolute lymphocyte count lab result (10^3/uL).',
    predictions.abs_neutrophil AS ABS_NEUTROPHIL WITH SYNONYMS = ('neutrophil count', 'ANC')
      COMMENT = 'Absolute neutrophil count lab result (10^3/uL).',
    predictions.albumin AS ALBUMIN COMMENT = 'Serum albumin lab result (g/dL). Low values indicate poor nutritional/inflammatory status.',
    predictions.bnp AS BNP WITH SYNONYMS = ('B-type natriuretic peptide', 'brain natriuretic peptide')
      COMMENT = 'BNP lab result (pg/mL); elevated values indicate heart failure.',
    predictions.bun AS BUN WITH SYNONYMS = ('blood urea nitrogen', 'urea')
      COMMENT = 'Blood urea nitrogen lab result (mg/dL); elevated values indicate renal dysfunction.',
    predictions.calcium AS CALCIUM COMMENT = 'Serum calcium lab result (mg/dL).',
    -- Prediction
    predictions.blood_culture_flag AS BLOOD_CULTURE COMMENT = 'Numeric blood culture result (1 = positive, 0 = negative).',
    predictions.codeblue_probability AS CODEBLUE_PROBABILITY
      WITH SYNONYMS = ('probability', 'code blue risk', 'predicted probability')
      COMMENT = 'Model-predicted probability (0-1) that the patient will experience a Code Blue event. Higher means higher risk.',
    predictions.risk_score_pct AS ROUND(CODEBLUE_PROBABILITY * 100, 1)
      WITH SYNONYMS = ('risk score', 'risk score percent', 'risk percentage', 'patient risk score')
      COMMENT = 'Patient risk score as a percentage (0-100): Code Blue probability x 100, rounded to 1 decimal.',
    predictions.actual_codeblue_flag AS CODEBLUE COMMENT = 'Numeric actual Code Blue outcome (1 = occurred, 0 = did not).',
    predictions.predicted_codeblue_flag AS CODEBLUE_PREDICTION COMMENT = 'Numeric predicted Code Blue outcome (1 = predicted, 0 = not predicted).'
  )
  DIMENSIONS (
    predictions.primary_mrn AS PRIMARYMRN WITH SYNONYMS = ('MRN', 'medical record number', 'patient id')
      COMMENT = 'Primary medical record number uniquely identifying the patient.',
    predictions.birth_date AS BIRTHDATE WITH SYNONYMS = ('date of birth', 'DOB') COMMENT = 'Patient date of birth.',
    predictions.gender AS GENDER WITH SYNONYMS = ('sex') COMMENT = 'Patient gender. Values: Male, Female.',
    predictions.age_group AS CASE WHEN AGE < 40 THEN '18-39' WHEN AGE < 65 THEN '40-64' WHEN AGE < 80 THEN '65-79' ELSE '80+' END
      WITH SYNONYMS = ('age band', 'age bracket') COMMENT = 'Age band of the patient. Values: 18-39, 40-64, 65-79, 80+.',
    predictions.entry_date AS ENTRY_DATE WITH SYNONYMS = ('admission date', 'admit date')
      COMMENT = 'Timestamp the patient was admitted to the hospital.',
    predictions.admission_month AS DATE_TRUNC('month', ENTRY_DATE) COMMENT = 'Month of admission, for monthly risk trends.',
    predictions.discharge_date AS DISCHARGE_DATE COMMENT = 'Timestamp the patient was discharged from the hospital.',
    predictions.blood_culture AS BLOOD_CULTURE WITH SYNONYMS = ('blood culture result', 'bacteremia')
      COMMENT = 'Blood culture result flag: 1 = positive, 0 = negative.',
    predictions.actual_codeblue AS IFF(CODEBLUE = 1, 'Yes', 'No') WITH SYNONYMS = ('actual outcome', 'had code blue', 'cardiac arrest')
      COMMENT = 'Whether the patient actually experienced a Code Blue event. Values: Yes, No.',
    predictions.predicted_codeblue AS IFF(CODEBLUE_PREDICTION = 1, 'Yes', 'No') WITH SYNONYMS = ('prediction', 'predicted outcome')
      COMMENT = 'Whether the model predicted a Code Blue event. Values: Yes, No.',
    predictions.risk_band AS CASE WHEN CODEBLUE_PROBABILITY >= 0.7 THEN 'High' WHEN CODEBLUE_PROBABILITY > 0.3 THEN 'Medium' ELSE 'Low' END
      WITH SYNONYMS = ('risk level', 'risk category', 'risk band', 'risk tier')
      COMMENT = 'Risk band from Code Blue probability: High (>= 0.7 / 70%), Medium (> 0.3 and < 0.7), Low (<= 0.3 / 30%). Use High for high-risk patient questions.',
    predictions.probability_bucket AS FLOOR(CODEBLUE_PROBABILITY * 10) / 10
      COMMENT = 'Code Blue probability rounded down to 0.1 buckets (0.0-0.9) for distribution analysis.',
    predictions.prediction_outcome AS CASE
        WHEN CODEBLUE = 1 AND CODEBLUE_PREDICTION = 1 THEN 'True Positive'
        WHEN CODEBLUE = 0 AND CODEBLUE_PREDICTION = 0 THEN 'True Negative'
        WHEN CODEBLUE = 0 AND CODEBLUE_PREDICTION = 1 THEN 'False Positive'
        ELSE 'False Negative' END
      WITH SYNONYMS = ('confusion matrix', 'prediction correctness')
      COMMENT = 'Comparison of predicted vs actual Code Blue: True Positive, True Negative, False Positive, False Negative.',
    predictions.model_version AS MODEL_VERSION WITH SYNONYMS = ('model', 'model name')
      COMMENT = 'Name/version of the ML model that produced the score, e.g. CODEBLUE_MODEL/V1_0.',
    predictions.scoring_timestamp AS SCORING_TIMESTAMP WITH SYNONYMS = ('scored at', 'prediction time')
      COMMENT = 'Timestamp when the model scored the patient.',
    predictions.top_risk_driver_1 AS EXPLAINABILITY_ATTRIBUTES:top_drivers[0]:feature::STRING
      WITH SYNONYMS = ('main risk factor', 'top driver', 'primary risk driver')
      COMMENT = 'Feature with the largest SHAP contribution for this patient (e.g. BUN, PULSE_OXIMETRY).',
    predictions.top_risk_driver_2 AS EXPLAINABILITY_ATTRIBUTES:top_drivers[1]:feature::STRING
      COMMENT = 'Feature with the second-largest SHAP contribution for this patient.',
    predictions.top_risk_driver_3 AS EXPLAINABILITY_ATTRIBUTES:top_drivers[2]:feature::STRING
      COMMENT = 'Feature with the third-largest SHAP contribution for this patient.',
    predictions.explanation_method AS EXPLAINABILITY_ATTRIBUTES:method::STRING
      COMMENT = 'Explainability method used to derive risk drivers (e.g. SHAP).',
    predictions.explainability_json AS TO_JSON(EXPLAINABILITY_ATTRIBUTES)
      COMMENT = 'SHAP explainability JSON with top_drivers (feature, value, contribution, direction increases/decreases risk).'
  )
  METRICS (
    predictions.patient_count AS COUNT(DISTINCT predictions.primary_mrn) WITH SYNONYMS = ('number of patients')
      COMMENT = 'Number of distinct patients.',
    predictions.high_risk_patient_count AS COUNT_IF(predictions.codeblue_probability >= 0.7)
      COMMENT = 'Number of High risk band patients (Code Blue probability >= 0.7).',
    predictions.avg_risk_score_pct AS AVG(predictions.risk_score_pct) WITH SYNONYMS = ('average risk score')
      COMMENT = 'Average patient risk score percentage (0-100).',
    predictions.avg_codeblue_probability AS AVG(predictions.codeblue_probability) WITH SYNONYMS = ('average probability')
      COMMENT = 'Average predicted Code Blue probability.',
    predictions.max_codeblue_probability AS MAX(predictions.codeblue_probability) COMMENT = 'Highest predicted Code Blue probability.',
    predictions.actual_codeblue_count AS SUM(predictions.actual_codeblue_flag) COMMENT = 'Number of actual Code Blue events.',
    predictions.predicted_codeblue_count AS SUM(predictions.predicted_codeblue_flag) COMMENT = 'Number of predicted Code Blue events.',
    predictions.actual_codeblue_rate AS AVG(predictions.actual_codeblue_flag) COMMENT = 'Share of patients with an actual Code Blue (0-1).',
    predictions.predicted_codeblue_rate AS AVG(predictions.predicted_codeblue_flag) COMMENT = 'Share of patients predicted to have a Code Blue (0-1).',
    predictions.model_accuracy AS AVG(IFF(predictions.actual_codeblue_flag = predictions.predicted_codeblue_flag, 1, 0))
      COMMENT = 'Fraction of patients where prediction matched actual outcome.',
    predictions.model_precision AS SUM(IFF(predictions.actual_codeblue_flag = 1 AND predictions.predicted_codeblue_flag = 1, 1, 0)) / NULLIF(SUM(predictions.predicted_codeblue_flag), 0)
      COMMENT = 'Precision: true positives / predicted positives.',
    predictions.model_recall AS SUM(IFF(predictions.actual_codeblue_flag = 1 AND predictions.predicted_codeblue_flag = 1, 1, 0)) / NULLIF(SUM(predictions.actual_codeblue_flag), 0)
      WITH SYNONYMS = ('sensitivity') COMMENT = 'Recall: true positives / actual positives.',
    predictions.avg_age AS AVG(predictions.age_years) COMMENT = 'Average patient age.',
    predictions.avg_length_of_stay AS AVG(predictions.length_of_stay_days) WITH SYNONYMS = ('ALOS') COMMENT = 'Average length of stay in days.',
    predictions.avg_systolic_bp AS AVG(predictions.systolic_bp) COMMENT = 'Average systolic blood pressure.',
    predictions.avg_pulse AS AVG(predictions.pulse) COMMENT = 'Average heart rate.',
    predictions.avg_pulse_oximetry AS AVG(predictions.pulse_oximetry) COMMENT = 'Average SpO2.',
    predictions.avg_respiration AS AVG(predictions.respiration) COMMENT = 'Average respiratory rate.',
    predictions.avg_bun AS AVG(predictions.bun) COMMENT = 'Average BUN.',
    predictions.avg_bnp AS AVG(predictions.bnp) COMMENT = 'Average BNP.',
    predictions.avg_albumin AS AVG(predictions.albumin) COMMENT = 'Average albumin.',
    predictions.positive_blood_culture_count AS SUM(predictions.blood_culture_flag) COMMENT = 'Number of patients with positive blood culture.'
  )
  COMMENT = 'Semantic view for Code Blue patient risk prediction analytics. This view combines patient demographic information, hospitalization details, flowsheet measurements, laboratory parameters, machine learning prediction outcomes, prediction probabilities, model metadata, and scoring information. The view enables clinicians, analysts, and operational teams to explore patient risk factors, identify high-risk patients, analyze prediction trends, and generate insights using natural language queries through Cortex Analyst.'
  AI_SQL_GENERATION 'High-risk patients means risk_band = High (codeblue_probability >= 0.7). Medium is > 0.3 and < 0.7; Low is <= 0.3. Show risk scores as risk_score_pct (0-100) unless the user asks for probability. Order high-risk patient lists by risk_score_pct descending.'
  AI_VERIFIED_QUERIES (
    top_risk_patients AS (
      QUESTION 'Which patients have the highest predicted Code Blue risk?'
      ONBOARDING_QUESTION TRUE
      SQL 'SELECT primary_mrn, risk_score_pct, risk_band, age_years, gender, top_risk_driver_1, top_risk_driver_2, top_risk_driver_3 FROM predictions WHERE risk_band = ''High'' ORDER BY risk_score_pct DESC NULLS LAST LIMIT 20'
    ),
    patients_prob_gt_80 AS (
      QUESTION 'How many patients have a prediction probability greater than 80%?'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT COUNT(DISTINCT primary_mrn) AS patient_count FROM predictions WHERE codeblue_probability > 0.8'
    ),
    high_risk_avg_vitals AS (
      QUESTION 'What are the average vital signs for high-risk patients?'
      ONBOARDING_QUESTION TRUE
      SQL 'SELECT COUNT(DISTINCT primary_mrn) AS patient_count, AVG(pulse) AS avg_pulse, AVG(systolic_bp) AS avg_systolic_bp, AVG(diastolic_bp) AS avg_diastolic_bp, AVG(respiration) AS avg_respiration, AVG(pulse_oximetry) AS avg_pulse_oximetry, AVG(temperature) AS avg_temperature, AVG(o2_flow_rate) AS avg_o2_flow_rate FROM predictions WHERE risk_band = ''High'''
    ),
    risk_by_age_group AS (
      QUESTION 'Which age groups have the highest Code Blue risk?'
      ONBOARDING_QUESTION TRUE
      SQL 'SELECT age_group, COUNT(DISTINCT primary_mrn) AS patient_count, AVG(risk_score_pct) AS avg_risk_score_pct, COUNT_IF(risk_band = ''High'') AS high_risk_patient_count, AVG(actual_codeblue_flag) AS actual_codeblue_rate FROM predictions GROUP BY age_group ORDER BY avg_risk_score_pct DESC NULLS LAST'
    ),
    predictions_by_gender AS (
      QUESTION 'Show Code Blue predictions by gender.'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT gender, COUNT(DISTINCT primary_mrn) AS patient_count, SUM(predicted_codeblue_flag) AS predicted_codeblue_count, SUM(actual_codeblue_flag) AS actual_codeblue_count, AVG(risk_score_pct) AS avg_risk_score_pct, COUNT_IF(risk_band = ''High'') AS high_risk_patient_count FROM predictions GROUP BY gender ORDER BY gender'
    ),
    high_risk_top_drivers AS (
      QUESTION 'What clinical factors are most common among high-risk patients?'
      ONBOARDING_QUESTION TRUE
      SQL 'SELECT top_risk_driver_1, COUNT(DISTINCT primary_mrn) AS patient_count, AVG(risk_score_pct) AS avg_risk_score_pct FROM predictions WHERE risk_band = ''High'' GROUP BY top_risk_driver_1 ORDER BY patient_count DESC NULLS LAST'
    ),
    high_risk_last_30_days AS (
      QUESTION 'How many patients were classified as high risk during the last 30 days?'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT COUNT(DISTINCT primary_mrn) AS high_risk_patient_count FROM predictions WHERE risk_band = ''High'' AND entry_date >= DATEADD(day, -30, CURRENT_DATE())'
    ),
    probability_distribution AS (
      QUESTION 'What is the distribution of prediction probabilities?'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT probability_bucket, COUNT(DISTINCT primary_mrn) AS patient_count FROM predictions GROUP BY probability_bucket ORDER BY probability_bucket'
    ),
    abnormal_bnp_high_risk AS (
      QUESTION 'Show patients with abnormal BNP and high Code Blue probability.'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT primary_mrn, bnp, risk_score_pct, risk_band, age_years, gender, top_risk_driver_1 FROM predictions WHERE bnp > 100 AND codeblue_probability >= 0.7 ORDER BY risk_score_pct DESC NULLS LAST'
    ),
    risk_across_age_categories AS (
      QUESTION 'Compare Code Blue risk across different age categories.'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT age_group, risk_band, COUNT(DISTINCT primary_mrn) AS patient_count, AVG(risk_score_pct) AS avg_risk_score_pct FROM predictions GROUP BY age_group, risk_band ORDER BY age_group, risk_band'
    ),
    model_performance AS (
      QUESTION 'How accurate is the Code Blue prediction model?'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT model_version, COUNT(DISTINCT primary_mrn) AS patient_count, AVG(IFF(actual_codeblue_flag = predicted_codeblue_flag, 1, 0)) AS model_accuracy, SUM(IFF(actual_codeblue_flag = 1 AND predicted_codeblue_flag = 1, 1, 0)) / NULLIF(SUM(predicted_codeblue_flag), 0) AS model_precision, SUM(IFF(actual_codeblue_flag = 1 AND predicted_codeblue_flag = 1, 1, 0)) / NULLIF(SUM(actual_codeblue_flag), 0) AS model_recall FROM predictions GROUP BY model_version'
    ),
    monthly_risk_trend AS (
      QUESTION 'What is the monthly trend of high-risk patients?'
      ONBOARDING_QUESTION FALSE
      SQL 'SELECT admission_month, COUNT(DISTINCT primary_mrn) AS patient_count, COUNT_IF(risk_band = ''High'') AS high_risk_patient_count, AVG(risk_score_pct) AS avg_risk_score_pct FROM predictions GROUP BY admission_month ORDER BY admission_month'
    )
  )