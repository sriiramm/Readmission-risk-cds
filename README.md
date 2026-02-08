🏥 30-Day Readmission Risk — Clinical Decision Support System (CDS)
End-to-end healthcare analytics project focused on cost-optimal, capacity-constrained decision making.

This project demonstrates how machine learning can be translated into real operational and financial decisions in healthcare. Instead of stopping at readmission risk prediction, I reframed the problem as a capacity-constrained intervention allocation challenge—selecting which patients to intervene on each day given limited staff and budget. Using real hospital data, cost-sensitive optimization, and operational simulation, the system identifies a staffing “sweet spot” that maximizes net savings while controlling false positives. Under conservative assumptions, the prototype estimates ~$1.7M in annual net savings for a mid-size hospital, with full transparency via SHAP explanations and fairness diagnostics. The project reflects how data science is applied in practice: balancing accuracy, cost, capacity, and equity.

📌 Problem Statement

Hospital readmissions within 30 days are costly, often penalized, and partially preventable.
While machine-learning models can predict readmission risk, probability alone does not drive real-world decisions.

Hospitals face operational constraints:

Limited care coordination staff

Fixed intervention budgets

Not all readmissions are avoidable

False positives create real costs and clinician fatigue

The real question is not “Who has high risk?” but:

“Given limited capacity, which patients should we intervene on to maximize net savings?”

This project reframes readmission prediction as a cost-sensitive, capacity-constrained resource allocation problem.

🎯 Project Objectives

Predict 30-day readmission risk using real hospital data

Translate model predictions into daily operational decisions

Optimize interventions based on cost, effectiveness, and staffing limits

Quantify financial impact and ROI

Provide model transparency (SHAP) and fairness diagnostics

Deliver insights via an interactive Streamlit dashboard

📊 Data

UCI Diabetes 130-US Hospitals Dataset (1999–2008)

~100,000 inpatient encounters

Demographics, diagnoses, procedures, medications, utilization history

Binary target: readmitted within 30 days

All preprocessing ensures:

No data leakage

Train/test separation

Preservation of fairness-relevant attributes (race, gender, age)

🧠 Modeling Approach
Baseline & Production Model

XGBoost classifier

Handles nonlinearity, interactions, and class imbalance

Evaluated using ROC-AUC (ranking quality)

Why probability thresholds were rejected

Traditional thresholds (e.g., p ≥ 0.5) produced:

Zero false positives

Unrealistically large savings

Poor recall of preventable cases

➡️ Rejected in favor of cost-optimal, Top-K decision logic

🔁 Decision Policy: Capacity-Constrained Top-K Allocation

Instead of thresholding probabilities:

Rank patients by predicted risk

Intervene on Top-K patients per day, where K = staff capacity

Optimize outcomes under real constraints

This reflects how hospitals actually operate.

⚙️ Operational Simulation

The system simulates daily hospital operations using Monte Carlo sampling:

Inputs

Daily discharges (avg)

Max interventions/day (staff capacity)

Cost per readmission

Cost per intervention

Intervention effectiveness

Avoidable fraction of readmissions (realism)

Outputs

Avg TP/day and FP/day

Precision at capacity (TP / (TP + FP))

Prevented readmissions/day

Net savings/day

Annualized net savings

ROI vs capacity curve (diminishing returns)

💰 Business Impact (Simulated Results)

Under conservative and explicit assumptions:

Assumption	Value
Daily discharges	100
Interventions/day (K)	40
Cost per readmission	$10,000
Cost per intervention	$300
Effectiveness	25%
Avoidable fraction	60%
Results

~11.1 true readmissions flagged per day

~1.7 readmissions prevented per day

Precision @ K=40: ~27.7%

Net savings/day: ~$4,620

Estimated annual net savings: ~$1.69M

The ROI curve reveals a clear “capacity sweet spot”:

Increasing capacity improves outcomes initially

Returns diminish as false positives grow

Over-scaling interventions can destroy value

➡️ Optimization should be driven by cost, not probability thresholds.

🔍 Explainability (SHAP)

For clinician and stakeholder trust:

Patient-level SHAP explanations

Top contributing features shown per patient

Enables transparent, auditable decision support

⚖️ Fairness & Ethics

The dashboard includes a fairness audit under the Top-K policy, reporting by subgroup:

Selection rate

True Positive Rate (selected | true readmission)

False Positive Rate (selected | no readmission)

This ensures:

Bias is monitored at the decision level, not just model level

Allocation impacts are visible before deployment

🖥️ Dashboard (Streamlit)

The interactive dashboard provides:

Executive-level ROI summary

Daily operations controls

Capacity vs ROI visualization

Patient-level explanations

Population-level risk distribution

Fairness diagnostics

🧠 Key Takeaways

High model accuracy ≠ high business value

Capacity constraints fundamentally change optimal decisions

Cost-sensitive optimization produces more realistic, defensible outcomes

Transparent assumptions build trust with stakeholders

Analytics must align with how organizations actually operate

🚀 Skills Demonstrated

Healthcare analytics

Cost-sensitive ML optimization

Operations & decision science

Monte Carlo simulation

XGBoost, SHAP

Streamlit app development

Fairness and ethical AI analysis

Business communication for technical systems
