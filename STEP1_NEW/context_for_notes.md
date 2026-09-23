============================================================
MASTER PROJECT CONTINUATION CONTEXT
FINAL YEAR RESEARCH PROJECT
============================================================

PROJECT TITLE
--------------
A Staged Framework for Data-Driven Prediction, Characterization,
Explainable Classification, and Decision Support for Air-Core
Behaviour in a Rotating System

MAIN PROJECT IDEA
-----------------
The project studies air-core behaviour in a rotating system using:

INPUTS:
1. Port diameter d (mm)
   Levels: 6, 8, 10, 12, 14, 16 mm
2. RPM
   Levels: 0, 20, 40, ..., 260 RPM

EXPERIMENTAL DESIGN:
6 diameters × 14 RPM levels = 84 real experimental observations.

CURRENT RESPONSE VARIABLES:
1. hc
2. A = area occupied by air core
3. Vmin = recorded/defined minimum air-core volume response

IMPORTANT:
Do NOT currently assume that Vmin = A × hc numerically.
Step 5A showed that the measured data do NOT support this identity.
Vmin must currently be treated as a separate response unless the
original experimental methodology/source establishes a valid formula,
conversion factor, or unit relationship.

PRIMARY RESEARCH AIM
--------------------
Develop a staged, data-driven framework that:
1. predicts air-core responses,
2. characterizes how diameter and RPM affect them,
3. investigates high-RPM extrapolation where experiments are limited,
4. performs engineering-relevant classification,
5. explains predictions/classification using XAI,
6. integrates all of this into a decision-support framework.

============================================================
CRITICAL METHODOLOGICAL RULES
============================================================

RULE 1 — MEASURED > INTERPOLATED > EXTRAPOLATED
-----------------------------------------------
Real measured observations have the strongest evidential status.

Interpolated values:
- generated inside the experimentally supported domain,
- model-generated,
- not measurements.

Extrapolated values:
- generated outside the observed experimental domain,
- engineering predictions only,
- must never be described as experimental measurements.

RULE 2 — FINAL TEST SET IS LOCKED
----------------------------------
84 real observations were split BEFORE interpolation/model development.

67 = Train_Anchors
17 = Test_Holdout

Random seed = 42

The 17 holdout observations must remain untouched for:
- model fitting,
- interpolation construction,
- ML training,
- hyperparameter tuning,
- model selection based on test performance.

RULE 3 — NO SYNTHETIC-DATA LEAKAGE
-----------------------------------
Do NOT:
- generate interpolation from all 84 points and then call the result
  independent validation,
- train ML on synthetic points and test on synthetic points generated
  from the same real dataset,
- use Test_Holdout information indirectly.

If interpolated data are used for training augmentation, the final
evaluation MUST still be on the same untouched 17 real experimental points.

RULE 4 — RESPONSE-SPECIFIC MODELS ARE ALLOWED
-----------------------------------------------
There is NO requirement that hc, A, and Vmin use the same mathematical
model.

Model selection must be response-specific and evidence-based.

RULE 5 — DO NOT CHOOSE A MODEL ONLY FROM R²
--------------------------------------------
Consider:
- R²
- RMSE
- MAE
- MSE
- MAPE
- coverage
- physical validity
- residual behaviour
- progressive stability
- model complexity
- engineering plausibility

RULE 6 — DO NOT HIDE FAILURES
------------------------------
Negative predictions, missing predictions, poor extrapolation,
coverage limitations, and failed consistency checks are useful research
findings and must be documented.

============================================================
DATA / FILE STRUCTURE
============================================================

PROJECT FOLDER
--------------
D:\Projects\clg_final_yr\STEP1_NEW

SOURCE
------
Original.xlsx

WORKING DATA COLUMNS
--------------------
Point_ID | d | Rpm | hc | A | Vmin

============================================================
COMPLETED WORK
============================================================

STEP 1 — DATA VALIDATION + LOCKED TRAIN/TEST SPLIT
---------------------------------------------------
Script:
Step1_Stratified_Split.py

Output:
Step1_Stratified_Split.xlsx

Results:
- 84 real observations
- 67 Train_Anchors
- 17 Test_Holdout
- holdout fraction = 20.24%
- seed = 42
- all six diameters represented
- holdout spread across RPM regions
- split is reproducible and space-covering, but NOT claimed as a
  mathematically optimal space-filling design.

LOCKED TEST POINT IDs:
71, 79, 83,
59, 64, 70,
43, 51, 53,
29, 36, 42,
18, 23, 27,
4, 8

These remain untouched until final validation stages.

Step 1 workbook contains:
Working_Data
Train_Anchors
Test_Holdout
Test_Point_IDs
Split_Log
Diameter_Check
RPM_Check

============================================================

STEP 2 — LOOCV METHOD SELECTION
--------------------------------
Script:
Step2_LOOCV_Method_Selection.py

Input:
Step1_Stratified_Split.xlsx
ONLY Train_Anchors used.

Methods tested:
1. Linear Griddata
2. Cubic Griddata
3. 1st-order RSM
4. 2nd-order RSM
5. 3rd-order RSM

Metrics:
R², RMSE, MAE, MSE, MAPE

Important:
Griddata had valid predictions for 62/67 LOOCV folds because some
held-out points were outside the convex hull.
RSM had 67/67 valid folds.

LOCKED STEP 2 RESULTS:

hc:
Cubic Griddata
R² = 0.995522
RMSE = 4.540914
MAE = 3.594157
MAPE = 2.9293%

A:
Cubic Griddata
R² = 0.994657
RMSE = 1.697695
MAE = 1.119234
MAPE = 5.5470%

Vmin:
3rd-order RSM
R² = 0.978755
RMSE = 0.014057
MAE = 0.010939
MAPE = 10.3064%

Output:
Step2_LOOCV_Results.xlsx

Sheets:
LOOCV_Summary
Best_Methods
Fold_Predictions

KEY RESEARCH FINDING:
Excellent interpolation performance does NOT automatically imply
good extrapolation performance.

============================================================

STEP 3 — RPM INTERPOLATION
---------------------------
Purpose:
Generate a denser RPM representation every 5 RPM while keeping diameter
fixed.

Important methodological decision:
2-D Griddata is NOT appropriate for a fixed-diameter RPM slice because
the slice is collinear in input space.

Therefore:
Interpolation is performed in 1-D RPM separately for each diameter.

FINAL INTERPOLATION METHODS:

hc:
1-D Cubic Spline

A:
1-D Cubic Spline

Vmin:
1-D PCHIP

Why PCHIP for Vmin:
Cubic spline produced negative synthetic Vmin values at low RPM.
PCHIP:
- removed the negative values,
- slightly improved interpolation validation.

STEP 3B VALIDATION:
55 valid interior-anchor folds per response.

hc:
R² = 0.984755
RMSE = 7.198547
MAE = 4.848695
MAPE = 4.1764%

A:
R² = 0.997310
RMSE = 1.227812
MAE = 0.576328
MAPE = 1.5866%

Vmin:
R² = 0.988530
RMSE = 0.008501
MAE = 0.005620
MAPE = 6.5024%

Boundary folds were excluded because holding out a boundary point would
turn the test into extrapolation rather than interpolation.

STEP 3C:
Compared Cubic Spline vs PCHIP for Vmin.

Cubic Spline:
R² = 0.988530
RMSE = 0.008501
MAE = 0.005620
MAPE = 6.5024%
Negative 5-RPM predictions = 3

PCHIP:
R² = 0.989072
RMSE = 0.008297
MAE = 0.005119
MAPE = 6.2650%
Negative 5-RPM predictions = 0

Negative cubic-spline points occurred at:
d=14, RPM=5
d=16, RPM=5
d=16, RPM=10

FINAL STEP 3D DATASET:
67 measured + 219 synthetic = 286 rows per response

Output:
Step3_Final_RPM_Interpolation.xlsx

Validation:
- 67 training anchors
- 219 synthetic points
- 286 total
- test leakage = 0
- Vmin negative values = 0

This is the locked RPM-interpolated training-domain dataset.

============================================================

STEP 4 — RPM EXTRAPOLATION
===========================

EXPERIMENTAL LIMITATION:
Measured RPM only goes up to 260 RPM.

Engineering motivation:
Real-world conditions may exceed the experimentally accessible RPM range,
and the project intends to investigate prediction beyond the measured range.

Therefore extrapolation is retained, but it must be separately validated.

============================================================

STEP 4A — INITIAL EXTRAPOLATION VALIDATION
--------------------------------------------
Highest 3 available training RPM anchors were withheld per diameter.

Initial RSM result:

hc:
3rd-order RSM
R² = -22.567
RMSE = 55.645
MAPE = 15.275%
=> unacceptable extrapolation

A:
best initial order = 1st order
R² = 0.642
RMSE = 14.957
MAPE = 95.882%
=> weak/unacceptable for direct high-RPM prediction

Vmin:
3rd-order RSM
R² = 0.886
RMSE = 0.01944
MAPE = 6.541%
=> promising

============================================================

STEP 4B — PROGRESSIVE EXTRAPOLATION VALIDATION
------------------------------------------------
Only 67 Train_Anchors were used.
17 final test points were NOT used.
219 interpolation points were NOT used.

Scenarios:

Scenario 1:
withhold highest 1 AVAILABLE training RPM per diameter
fit = 61
holdout = 6

Scenario 2:
withhold highest 2 AVAILABLE training RPM per diameter
fit = 55
holdout = 12

Scenario 3:
withhold highest 3 AVAILABLE training RPM per diameter
fit = 49
holdout = 18

WHY "AVAILABLE":
Some high-RPM nominal points were already part of the locked experimental
holdout. Therefore scenarios use the highest RPM actually available in
Train_Anchors.

RESULTS:

hc:
Scenario 1 best = 3rd RSM
R² = -14.696
RMSE = 30.209
MAPE = 8.222%

Scenario 2 best = 3rd RSM
R² = -20.974
RMSE = 41.772
MAPE = 10.501%

Scenario 3 best = 2nd RSM
R² = -26.134
RMSE = 59.707
MAPE = 18.675%

Decision:
Polynomial RSM extrapolation for hc is NOT scientifically defensible.

A:
Scenario 1 best = 1st RSM
R² = 0.840
RMSE = 10.126
MAPE = 63.045%

Scenario 2 best = 1st RSM
R² = 0.778
RMSE = 11.842
MAPE = 74.418%

Scenario 3 best = 1st RSM
R² = 0.642
RMSE = 14.957
MAPE = 95.882%

Decision:
Polynomial RSM extrapolation for A is weak and degrades with
extrapolation depth.

Vmin:
Scenario 1:
3rd RSM
R² = 0.933
RMSE = 0.015431
MAPE = 5.182%

Scenario 2:
3rd RSM
R² = 0.910
RMSE = 0.017680
MAPE = 5.757%

Scenario 3:
3rd RSM
R² = 0.886
RMSE = 0.019450
MAPE = 6.935%

Decision:
3rd-order RSM is the leading extrapolation model for Vmin.

============================================================

STEP 4C — ALTERNATIVE EXTRAPOLATION MODELS
-------------------------------------------
Purpose:
Find better-behaved extrapolation functions for hc and A.

Candidate models:
1. Linear
2. Quadratic
3. Cubic
4. Logarithmic
5. Square-root
6. Saturating

Progressive scenarios:
highest 1, highest 2, highest 3 AVAILABLE RPM anchors withheld.

RESULT:

SATURATING MODEL CLEARLY WON FOR BOTH hc AND A.

A:

Scenario 1:
R² = 0.999817
RMSE = 0.342469
MAPE = 1.061024%

Scenario 2:
R² = 0.999464
RMSE = 0.581692
MAPE = 1.147910%

Scenario 3:
R² = 0.998817
RMSE = 0.859877
MAPE = 1.382929%

A remained physically nonnegative.

hc:

Scenario 1:
R² = 0.650761
RMSE = 4.506037
MAPE = 1.291177%

Scenario 2:
R² = 0.697523
RMSE = 4.900978
MAPE = 1.340537%

Scenario 3:
R² = 0.746428
RMSE = 5.771983
MAPE = 1.784476%

Saturating model was much better than alternatives, but hc is NOT
"perfectly validated".
It is the best available/stable candidate, but still has substantial
remaining uncertainty.

Physical checks:
No negative hc or A predictions in the progressive Step 4C validation.

IMPORTANT FINAL EXTRAPOLATION DECISION:
hc -> Saturating RPM model
A -> Saturating RPM model
Vmin -> 3rd-order RSM

This is an extrapolation-specific model choice.
It does NOT replace the in-domain interpolation models.

============================================================

STEP 4D — FINAL HIGH-RPM EXTRAPOLATION
---------------------------------------
Output:
Step4D_Final_RPM_Extrapolation.xlsx

Generated:
Diameters = 6, 8, 10, 12, 14, 16 mm
RPM = 280, 285, 290, 295, 300, 305, 310, 315, 320

Total:
6 × 9 = 54 extrapolated points

No test-coordinate overlap.

Methods:
hc = Saturating
A = Saturating
Vmin = 3rd-order RSM

Physical checks:

hc_pred:
54 points
0 nonfinite
0 negative
minimum = 281.872586
maximum = 302.007619

A_pred:
54 points
0 nonfinite
0 negative
minimum = 5.210937
maximum = 79.974516

Vmin_pred:
54 points
0 nonfinite
0 negative
minimum = 0.187517
maximum = 0.398433

Grid:
54 expected
54 actual
6 diameters
9 RPM levels per diameter
280–320 RPM
coordinate overlap = 0
complete grid = TRUE

IMPORTANT:
These are engineering model estimates only.
They are NOT experimental observations.

============================================================

STEP 5 — FINAL EXPERIMENTAL HOLDOUT VALIDATION
-----------------------------------------------
Script:
Step5_Final_Holdout_Validation.py

Input:
Step1_Stratified_Split.xlsx

Training:
67

Holdout:
17

Training/test Point_ID overlap:
0

Training/test coordinate overlap:
0

MODELS TESTED:

hc:
1. Cubic Griddata
2. Saturating RPM model

A:
1. Cubic Griddata
2. Saturating RPM model

Vmin:
1. 3rd-order RSM

FINAL HOLDOUT RESULTS:

hc:
Cubic Griddata:
R² = 0.999245
RMSE = 2.441890
MAE = 1.844882
MAPE = 1.440474%
Coverage = 16/17 = 94.12%

Saturating:
R² = 0.982591
RMSE = 13.033561
MAE = 8.149799
MAPE = 44.405441%
Coverage = 17/17

Conclusion:
Cubic Griddata is much better for IN-DOMAIN prediction.
Saturating is retained for HIGH-RPM EXTRAPOLATION, not because it is
better in-domain but because extrapolation requires a functional form.

A:
Cubic Griddata:
R² = 0.998538
RMSE = 0.919840
MAE = 0.688520
MAPE = 3.588386%
Coverage = 16/17

Saturating:
R² = 0.973274
RMSE = 3.979626
MAE = 1.825827
MAPE = 1.032568%
Coverage = 17/17

Conclusion:
Cubic Griddata is better for IN-DOMAIN prediction.
Saturating remains the separately validated HIGH-RPM extrapolation model.

Vmin:
3rd-order RSM:
R² = 0.987684
RMSE = 0.011767
MAE = 0.009882
MAPE = 5.012270%
Coverage = 17/17

This is strong independent holdout performance.

Physical holdout issues:
hc Saturating = 1 negative prediction
A Cubic = 1 negative prediction
A Saturating = 3 negative predictions
Vmin 3rd RSM = 1 negative prediction

These physical-validity issues must be documented and investigated before
using models as unrestricted engineering predictors.

============================================================

STEP 5A — VMIN / A / hc CONSISTENCY INVESTIGATION
--------------------------------------------------
Purpose:
Determine whether the recorded Vmin really equals A × hc.

Script:
Step5A_Vmin_Consistency_Investigation.py

Measured 84-point diagnostic:

Vmin / (A*hc)

All 84:
78 valid nonzero-ratio observations
mean = 0.000049
median = 0.000036
std = 0.000035
min = 0.000017
max = 0.000118
CV = 71.414%

Training:
CV = 71.914%

Holdout:
CV = 71.630%

No observations were within 5% of the median ratio.
No observations were within 10% of the median ratio.

Constant-factor test:
Vmin = k*(A*hc)

Estimated k = 0.000023

R² = -0.306734
RMSE = 0.097311
MAE = 0.083047

=> poor fit

Log-log relationship:
slope = 0.458884

A proportional relationship would be expected to have slope near 1.
The observed slope is far from 1.

FINAL STEP 5A CONCLUSION:
The measured data do NOT support a stable constant-factor identity
between Vmin and A*hc.

Therefore:
DO NOT use:
Vmin = A × hc

as a mathematical identity unless the original experimental methodology
establishes a valid relationship/unit conversion.

The current scientifically defensible treatment is:
hc, A, and Vmin are separate responses.

VERY IMPORTANT:
The enormous A_pred × hc_pred values observed in Step 5 do NOT automatically
mean that the hc and A models are catastrophically wrong.
They demonstrate that multiplying the recorded A and hc columns is not a
valid way to reconstruct the recorded Vmin.

============================================================
CURRENT LOCKED MODEL STRUCTURE
============================================================

IN-DOMAIN / EXPERIMENTALLY SUPPORTED PREDICTION:

hc:
Cubic Griddata

A:
Cubic Griddata

Vmin:
3rd-order RSM

HIGH-RPM EXTRAPOLATION:

hc:
Saturating RPM model

A:
Saturating RPM model

Vmin:
3rd-order RSM

RPM INTERPOLATION DATASET:

hc:
1-D Cubic Spline

A:
1-D Cubic Spline

Vmin:
1-D PCHIP

============================================================
CURRENT MAJOR RESEARCH FINDINGS
============================================================

1. High interpolation accuracy does NOT guarantee high extrapolation
   accuracy.

2. hc and A require locally adaptive/in-domain interpolation but their
   polynomial RSM extrapolations were poor.

3. Saturating functional forms are much more stable than polynomial
   forms for hc/A high-RPM extrapolation.

4. Vmin behaves more smoothly under third-order RSM extrapolation.

5. PCHIP is preferable to cubic spline for Vmin interpolation because
   it avoids physically impossible negative interpolated values while
   slightly improving validation.

6. Different responses can legitimately use different modelling methods.

7. Vmin cannot currently be treated as numerically identical to A*hc.

8. The 17-point experimental holdout is the main independent validation
   evidence.

9. Interpolated and extrapolated data must never be presented as
   measurements.

10. The classification stage must NOT be invented before continuous
    response modelling is established.

============================================================
NEXT WORK — DO NOT SKIP THESE STEPS
============================================================

STEP 5B — SOURCE / RESPONSE DEFINITION AUDIT
=============================================

PURPOSE:
Resolve the meaning/units/definition of:
hc
A
Vmin

BEFORE ML, determine from the original experimental methodology/source:

1. Exact definition of hc
2. Exact definition of A
3. Exact definition of Vmin
4. Units of hc
5. Units of A
6. Units of Vmin
7. Whether Vmin is:
   - directly measured,
   - geometrically calculated,
   - scaled,
   - normalized,
   - converted,
   - or derived from another quantity
8. Whether a published/source equation connects Vmin to A and hc
9. Whether there is a missing geometric coefficient or conversion factor
10. Whether A is truly an area in the same dimensional sense assumed
    by the current calculations

OUTPUT:
Step5B_Response_Definition_Audit.xlsx

RECOMMENDED SHEETS:
- Source_Definitions
- Units
- Formula_Audit
- Variable_Roles
- Evidence
- Final_Decision

FINAL DECISION:
Either:
A. Source confirms a formula/conversion relationship
OR
B. No valid identity can be established, therefore hc/A/Vmin remain
   separate response variables.

DO NOT alter historical results.
Only clarify the interpretation.

============================================================

STEP 5C — CORRECT/LOCK RESPONSE DEFINITIONS
==============================================

Update the master progress document so that the old wording:

"Vmin = A × hc"

is removed/replaced.

Use wording conceptually equivalent to:

"Vmin is retained as an independently defined/recorded response.
A numerical identity between the recorded Vmin, A and hc values was tested
and was not supported by the measured data. Any physical relationship must
be established from the original experimental definition or methodology."

Also document the Step 5A failed consistency check as a research finding.

OUTPUT:
Updated progress document.

============================================================

STEP 6 — ML REGRESSION DATASET STRATEGY
========================================

Before fitting ML, create a controlled dataset strategy.

Prepare three conceptual datasets:

DATASET 1:
67 real training observations

DATASET 2:
67 real + 219 interpolated observations
= 286 rows

DATASET 3:
54 extrapolated engineering prediction points

RULE:
Dataset 3 is NOT a normal ML training set.

The 54 extrapolated points are predictions, not measurements.

STEP 6A should document exactly which dataset is used for which analysis.

============================================================

STEP 6A — ML BASELINE
======================

Before machine learning, establish mathematical baselines.

Baseline models:

hc:
Cubic Griddata for in-domain

A:
Cubic Griddata for in-domain

Vmin:
3rd-order RSM

Use these as benchmark models.

Question:
Does ML genuinely improve prediction over the validated mathematical
baseline?

============================================================

STEP 6B — ML MODEL CANDIDATES
==============================

Evaluate suitable models:

1. Random Forest Regressor
2. Gradient Boosting / XGBoost
3. Support Vector Regression
4. Artificial Neural Network

Do not assume ANN is best.

With only 67 real observations and 2 inputs, model complexity must be
controlled carefully.

============================================================

STEP 6C — ML TRAINING PROTOCOL
===============================

For each target:
hc
A
Vmin

Use:

67 real training observations

Perform:
- scaling where needed,
- model fitting,
- internal cross-validation,
- hyperparameter optimisation.

Do NOT tune against the 17 holdout observations.

Correct sequence:

67 training observations
       ↓
internal CV / hyperparameter tuning
       ↓
best model/configuration
       ↓
refit on full 67 training points
       ↓
predict 17 untouched test points

============================================================

STEP 6D — ML HYPERPARAMETER OPTIMISATION
==========================================

For each ML algorithm define a controlled search space.

Examples:

Random Forest:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

Gradient Boosting/XGBoost:
- number of estimators
- learning rate
- tree depth
- subsampling
- regularisation

SVR:
- C
- gamma
- epsilon
- kernel

ANN:
- hidden layer count
- neurons per layer
- activation
- learning rate
- regularisation
- early stopping

Use CV inside training only.

============================================================

STEP 6E — REAL-ONLY VS INTERPOLATED-AUGMENTED ML
================================================

This is an important research experiment.

EXPERIMENT A:
Train ML on 67 real observations only.

EXPERIMENT B:
Train ML on 67 real + 219 interpolated observations.

For BOTH:
Final evaluation = same 17 untouched real experiments.

Purpose:
Determine whether interpolation improves actual experimental
generalisation or merely teaches ML to reproduce the interpolation surface.

This comparison should become an important thesis result.

============================================================

STEP 6F — ML PERFORMANCE EVALUATION
====================================

For every model and every response calculate:

R²
RMSE
MAE
MSE
MAPE
Explained Variance
Adjusted R² where appropriate

Also calculate:
coverage
physical validity

Generate:
1. Actual vs predicted parity plot
2. Residual plot
3. Error distribution
4. Prediction vs RPM
5. Prediction vs diameter

============================================================

STEP 6G — ML MODEL SELECTION
=============================

Select final continuous model separately for:

hc
A
Vmin

Selection criteria:

1. 17-point real experimental performance
2. Training CV stability
3. Physical validity
4. Residual quality
5. Generalisation
6. Complexity
7. Meaningful improvement over baseline

Do NOT select solely by R².

Possible outcome:
hc = Model A
A = Model B
Vmin = RSM or Model C

That is acceptable.

============================================================

STEP 7 — RESIDUAL / ERROR ANALYSIS
===================================

For final models:

Check:
- bias
- systematic overprediction
- systematic underprediction
- heteroscedasticity
- outliers
- error vs RPM
- error vs diameter

Also calculate metrics separately by diameter:
6
8
10
12
14
16 mm

and preferably by RPM regions.

This is especially important near the high-RPM boundary.

============================================================

STEP 8 — AIR-CORE CHARACTERIZATION
===================================

This is where "Characterization" in the thesis title becomes substantive.

Use final validated continuous models to study:

1. hc as a function of d and RPM
2. A as a function of d and RPM
3. Vmin as a function of d and RPM

Generate response surfaces / contour plots where appropriate.

Study:
- RPM effect
- diameter effect
- interaction effect
- regions of rapid response change
- regions of stable response
- high-RPM trends

============================================================

STEP 8A — SENSITIVITY ANALYSIS
===============================

Quantify sensitivity to:

RPM
Diameter

Where appropriate calculate:
∂y/∂RPM
∂y/∂d

For ML models use:
- numerical sensitivity,
- permutation-based analysis,
- partial dependence,
- SHAP where appropriate.

Questions:
- Which input dominates?
- Does importance change by response?
- Are there diminishing returns?
- Where are critical/sensitive operating regions?

============================================================

STEP 8B — EXTRAPOLATION ANALYSIS
=================================

Use Step4D 54-point high-RPM dataset.

Analyse:
260 → 320 RPM trend

For each:
hc
A
Vmin

Check:
- monotonicity where physically expected
- saturation behaviour
- response stability
- comparison with observed high-RPM trend
- sensitivity to chosen extrapolation model

DO NOT call extrapolated points experimental.

Clearly label:
"Model-based engineering extrapolation."

============================================================

STEP 8C — EXTRAPOLATION UNCERTAINTY / APPLICABILITY
=====================================================

For high-RPM predictions document:

1. pseudo-extrapolation validation performance
2. distance beyond measured range
3. model stability
4. physical plausibility
5. uncertainty if feasible

Especially for hc:
state that the saturating model is the best validated candidate,
but extrapolation uncertainty remains material.

Do NOT overclaim 320 RPM confidence.

============================================================

STEP 9 — ENGINEERING CLASSIFICATION DEFINITION
================================================

DO NOT classify before continuous prediction is established.

First determine what "class" physically means.

Possible conceptual classes could be:
- low air-core condition
- moderate air-core condition
- high air-core condition
- critical/severe condition

BUT:
DO NOT choose classes merely because they produce good ML metrics.

The criterion must come from:
- experimental evidence,
- engineering requirements,
- established literature,
- operational limits,
- geometric constraints,
- or another defensible engineering rule.

The classification definition must be documented BEFORE classification
model training.

============================================================

STEP 9A — CREATE CLASS LABELS
==============================

For every valid operating condition:

Input:
d
RPM

Prediction/measurement:
relevant air-core response(s)

Engineering criterion:
threshold / rule

Output:
Class label

Keep:
- original measured response
- model prediction
- class
together for traceability.

============================================================

STEP 9B — CHECK CLASS DISTRIBUTION
==================================

Before fitting classification models:

Check:
- class counts
- class percentages
- class imbalance
- class overlap
- borderline cases

If imbalanced:
use appropriate training-only class-balancing techniques.

Do NOT rebalance the final untouched test set.

============================================================

STEP 10 — CLASSIFICATION MODELLING
===================================

Candidate classifiers:

1. Random Forest Classifier
2. Gradient Boosting / XGBoost
3. SVM classifier
4. other simple interpretable classifier if appropriate

Potential features:
- d
- RPM
- selected validated continuous predicted responses
depending on the final engineering logic.

Avoid feature leakage.

If class labels are created from a response:
do not accidentally use the same response in a way that trivially leaks
the answer unless that is explicitly part of the intended decision model.

============================================================

STEP 10A — CLASSIFICATION HYPERPARAMETER TUNING
================================================

Use only training data for:
- tuning
- balancing
- feature selection

Possible techniques:
- Grid Search
- Random Search
- controlled CV

Final independent evaluation:
real held-out observations where the final classification structure
permits direct test labelling.

============================================================

STEP 10B — CLASSIFICATION VALIDATION
=====================================

Evaluate:

Accuracy
Precision
Recall
F1-score
Balanced Accuracy where appropriate
Confusion Matrix
Class-wise performance

Also inspect:
- false positives
- false negatives
- boundary cases

In an engineering context, false negatives may be particularly important
if they correspond to missing a severe/critical air-core condition.

============================================================

STEP 11 — XAI
=============

Purpose:
Explain WHY the model predicts a class.

Use:
- SHAP
- permutation importance
- feature importance
- local explanations
- global explanations

Two levels:

GLOBAL:
What drives classification overall?

LOCAL:
Why did this specific operating condition get this class?

Example structure:

Input:
d = ___
RPM = ___

Predicted class:
___

Main contributors:
1. RPM
2. Diameter
3. predicted hc
4. predicted A
etc.

Do not invent explanations that the model does not support.

============================================================

STEP 11A — XAI FOR CONTINUOUS MODELS
=====================================

Also explain the regression models where useful.

For each final ML model:
- global feature importance
- SHAP summary
- individual prediction explanation

Compare:
- hc drivers
- A drivers
- Vmin drivers

This supports the "Data-Driven Prediction" and "Characterization"
parts of the thesis.

============================================================

STEP 12 — DECISION-SUPPORT FRAMEWORK
=====================================

Final architecture:

USER INPUT
    ↓
Port Diameter + RPM
    ↓
Continuous prediction
    ↓
hc + A + Vmin
    ↓
Engineering characterization
    ↓
Operating-condition classification
    ↓
XAI explanation
    ↓
Engineering interpretation
    ↓
Decision-support output

Possible output:

Input:
d = ___ mm
RPM = ___

Predicted:
hc = ___
A = ___
Vmin = ___

Classification:
__________

Dominant factors:
__________

Engineering interpretation:
__________

Applicability:
experimentally validated / interpolated / extrapolated

============================================================

STEP 12A — DECISION-SUPPORT RULES
==================================

The final system must know whether its input is:

1. inside measured experimental points,
2. inside experimentally supported interpolation region,
3. outside measured RPM but inside approved extrapolation range,
4. outside validated scope.

Example conceptual status:

VALIDATED EXPERIMENTAL REGION
6–16 mm
0–260 RPM

INTERPOLATED REGION
Inside experimental envelope but between measured RPM values.

ENGINEERING EXTRAPOLATION REGION
280–320 RPM
model-based only

OUTSIDE VALIDATED SCOPE
Any input outside the approved domain/range.

Do NOT present predictions outside validated scope without warning.

============================================================

STEP 12B — DECISION-SUPPORT IMPLEMENTATION
============================================

Later determine the most appropriate implementation:

Possible:
- Python application
- Streamlit interface
- Excel decision tool
- lightweight web interface

Do not build the interface too early.

First lock:
- models
- thresholds
- explanations
- applicability limits

Then build the interface.

============================================================

STEP 13 — FINAL ENGINEERING VALIDATION
=======================================

Combine:

1. 84 real experiments
2. 67 training observations
3. 17 independent test observations
4. 286-row interpolation dataset
5. 54-row high-RPM extrapolation dataset
6. final ML model results
7. classification results
8. XAI results

Create a final validation table.

Distinguish clearly:

MEASURED
INTERPOLATED
EXTRAPOLATED
ML PREDICTION
CLASSIFICATION

============================================================

STEP 13A — ROBUSTNESS ANALYSIS
===============================

Check:
- model stability
- repeated CV if appropriate
- sensitivity to split
- feature influence
- residual robustness
- physical validity
- extrapolation robustness
- classification robustness

Do not let the conclusion depend on one lucky split if additional
robustness analysis is feasible.

The fixed 17-point split remains the primary locked independent test;
additional robustness analyses must not overwrite it.

============================================================

STEP 14 — FINAL APPLICABILITY LIMITS
====================================

Explicitly define:

Experimentally validated:
d = 6–16 mm
RPM = 0–260 RPM

Interpolation:
inside the experimentally supported domain.

Extrapolation:
280–320 RPM
only with the separately validated extrapolation models.

Anything beyond:
NOT VALIDATED.

Document limitations:
- limited RPM range
- only six diameter levels
- relatively small real dataset
- model-dependent extrapolation
- uncertain hc extrapolation
- Vmin definition/relationship issue
- any classification-threshold limitations

============================================================

STEP 15 — FINAL RESEARCH QUESTIONS
===================================

The final thesis should answer:

RQ1:
Can air-core behaviour be predicted from port diameter and RPM?

RQ2:
How do diameter and RPM influence hc, A and Vmin?

RQ3:
Does machine learning outperform validated mathematical baselines?

RQ4:
Can validated functional extrapolation extend prediction beyond the
experimental RPM limit?

RQ5:
Can operating conditions be classified using an engineering-defensible
criterion?

RQ6:
Can XAI explain the major factors behind predictions/classification?

RQ7:
Can these components be integrated into a practical decision-support
framework?

============================================================

STEP 16 — THESIS DOCUMENTATION
================================

FINAL THESIS STRUCTURE SHOULD EVENTUALLY COVER:

1. Introduction
2. Literature Review
3. Experimental System / Dataset
4. Methodology
5. Data Validation
6. Train/Test Strategy
7. Interpolation
8. Extrapolation
9. Continuous Mathematical Modelling
10. Machine Learning
11. Model Evaluation
12. Air-Core Characterization
13. Classification
14. Explainable AI
15. Decision-Support Framework
16. Engineering Validation
17. Limitations
18. Conclusions
19. Future Work

============================================================
IMPORTANT FILES TO MAINTAIN
============================================================

Source:
Original.xlsx

Step 1:
Step1_Stratified_Split.py
Step1_Stratified_Split.xlsx

Step 2:
Step2_LOOCV_Method_Selection.py
Step2_LOOCV_Results.xlsx

Step 3:
Step3A_RPM_Interpolation.py
Step3B_RPM_Interpolation_Validity.py
Step3C_Vmin_Cubic_vs_PCHIP.py
Step3D_Final_RPM_Interpolation.py
Step3_Final_RPM_Interpolation.xlsx

Step 4:
Step4A_RPM_Extrapolation_Validation.py
Step4B_Progressive_RPM_Extrapolation_Validation.py
Step4B_Progressive_RPM_Extrapolation_Validation.xlsx
Step4C_Alternative_RPM_Extrapolation_Validation.py
Step4C_Alternative_RPM_Extrapolation_Validation.xlsx
Step4D_Final_RPM_Extrapolation.py
Step4D_Final_RPM_Extrapolation.xlsx

Step 5:
Step5_Final_Holdout_Validation.py
Step5_Final_Holdout_Validation.xlsx

Step 5A:
Step5A_Vmin_Consistency_Investigation.py
Step5A_Vmin_Consistency_Investigation.xlsx

NEXT FILES TO CREATE:
Step5B_Response_Definition_Audit.py
Step5B_Response_Definition_Audit.xlsx

Then ML files should follow a clean naming structure:
Step6A_ML_Dataset_Strategy.py
Step6B_ML_Baseline_Comparison.py
Step6C_ML_RandomForest.py
Step6D_ML_GradientBoosting.py
Step6E_ML_SVR.py
Step6F_ML_ANN.py
Step6G_ML_Hyperparameter_Optimization.py
Step6H_ML_Real_vs_Interpolated_Augmentation.py
Step7_Final_Continuous_Model_Selection.py
Step7A_Residual_Analysis.py
Step8_AirCore_Characterization.py
Step8A_Sensitivity_Analysis.py
Step8B_HighRPM_Engineering_Analysis.py
Step9_Classification_Definition.py
Step10_Classification_Modeling.py
Step10A_Classification_Validation.py
Step11_XAI.py
Step12_Decision_Support_Framework.py
Step13_Final_Engineering_Validation.py
Step14_Applicability_Robustness.py

Do NOT create all of these at once.
Create and run one stage at a time, inspect results, then lock the decision.

============================================================
CURRENT STATUS
============================================================

COMPLETED:
Step 1
Step 2
Step 3A
Step 3B
Step 3C
Step 3D
Step 4A
Step 4B
Step 4C
Step 4D
Step 5
Step 5A

CURRENT POSITION:
Step 5A completed.

IMMEDIATE NEXT STEP:
Step 5B — Source / Response Definition / Units Audit

THEN:
Step 5C — Correct and lock the definitions in the master document

THEN:
Step 6A — ML dataset strategy

THEN:
Step 6 — ML regression comparison

DO NOT jump to classification yet.

============================================================
IF STARTING A NEW CHAT
============================================================

Paste this entire document first.

Then say:

"I am continuing my final-year research project.
Treat the above as the locked project context.
We are currently at Step 5B.
Do NOT redo completed steps.
Do NOT change locked decisions unless new evidence requires it.
Start by checking what information is needed for Step 5B and then
continue one reproducible Python/Excel step at a time."

============================================================
NON-NEGOTIABLE SCIENTIFIC PRINCIPLES
============================================================

1. Never use the 17 holdout points for model development.
2. Never call synthetic interpolation values experimental.
3. Never call high-RPM extrapolated values measurements.
4. Never use synthetic test-on-synthetic performance as independent validation.
5. Do not force the same model onto every response.
6. Do not select models using R² alone.
7. Do not assume Vmin = A × hc without source confirmation.
8. Do not hide negative predictions or failed validations.
9. Do not define classification classes merely to improve ML scores.
10. Do not overclaim extrapolation reliability.
11. Keep complete scripts and Excel outputs for every stage.
12. Preserve Point_IDs.
13. Use BASE_DIR = Path(__file__).resolve().parent in scripts.
14. Every new step should have:
    - input file
    - exact method
    - validation logic
    - output workbook
    - checks
    - console summary
    - documented decision
15. Every model-selection decision must be traceable to numerical evidence.
16. The final decision-support framework must distinguish validated,
    interpolated, extrapolated, and outside-scope predictions.

============================================================
END OF MASTER CONTINUATION CONTEXT
============================================================