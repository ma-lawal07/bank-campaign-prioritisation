# Bank Campaign Prioritisation

Explainable customer targeting under a limited contact budget.

## Business question

With capacity to contact only a fraction of customers, which records should a bank prioritise?

This project compares logistic regression, random forest and gradient boosting to predict historical term-deposit subscriptions. A Streamlit demo ranks records, simulates contact budgets, explains prediction sensitivity and flags records for review.

## Dataset

Use the original **bank-full.csv**, containing 45,211 records, from [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing).

Download and extract `bank.zip`, then place `bank-full.csv` in `data/raw/`.

Source: Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

The dataset is distributed under CC BY 4.0.

## Setup

Run from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m pytest -q
python -m streamlit run app.py
```

Training recreates model files and evaluation results in `artifacts/`.

## Decision point and leakage prevention

The simulated decision happens before the first contact in a new campaign.

Inputs are age, job, marital status, education, credit default, balance, housing loan and personal loan.

Call duration is excluded because it is known after the call. Recorded contact channel, day, month and current-campaign contact count are also excluded. Previous-campaign fields are omitted from this initial version. Subscription outcome `y` is used only as the training and evaluation target.

The demo assumes the selected customer attributes were available at the decision point; the dataset does not fully verify their historical availability.

## Evaluation method

Records retain their original order:

- First 60%: model training.
- Next 20%: validation and development.
- Final 20%: held-out test evaluation.

Preprocessing is fitted only on training data. Random forest was chosen using validation ranking performance. Sigmoid calibration was fitted on the first half of validation and assessed on the second half.

The assessment is a development check because validation had already influenced model selection. The model choice was fixed before final test evaluation.

Budgets use the floor of record count multiplied by contact capacity. Tied predictions retain input order.

Metrics include precision and recall at the contact budget, lift over random selection, Brier score and calibration plots.

## Main results

At a 10% budget on the test period, the chosen calibrated random forest:

- Selected 904 records.
- Captured 384 historical subscribers.
- Achieved 42.48% precision and 13.45% recall.
- Achieved 1.34× lift over random selection.
- Had a Brier score of 0.2542, compared with 0.2607 for a constant probability based on the calibration-period subscription rate.

Logistic regression performed better at ranking on the final test period: 46.68% precision and 1.48× lift. This reversal shows that the validation winner did not remain the strongest ranking model in a later period.

Calibration improved probability error but substantial underestimation remained.

## Explanations and review flags

The exploration notebook uses permutation importance to assess which inputs support precision at the 10% budget.

The demo provides individual sensitivity checks by replacing one feature at a time with its training median or most common category. These effects are not additive feature contributions and can involve unrealistic combinations.

Review flags identify large ranking disagreements between model families and records close to the contact-budget boundary. They are illustrative heuristics, not confidence intervals.

## Business recommendation

The historical results support investigating probability ranking as a way to allocate limited contact capacity. The chosen model concentrated historical subscribers above the random-selection expectation.

The prototype is not ready for operational use. Its ranking advantage weakened in the test period, and probability estimates remained poorly calibrated. Prospective validation, monitoring and fairness assessment would be needed before deployment.

## Limitations

- Subscription rates changed substantially between training, validation and test periods.
- Historical campaign records may not represent a future customer population.
- Record identifiers do not establish unique customers; repeated customers cannot be reliably grouped with the available identifiers.
- Demographic inputs may produce unequal selection patterns; fairness has not been established.
- Sensitivity explanations describe model behaviour, not causal effects.
- The data cannot establish whether contacting a customer causes subscription.
- Historical subscriber capture does not establish additional revenue or campaign profitability.