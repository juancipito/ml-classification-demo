# ML Academic Classification Demo

A reproducible classification workflow on a fully synthetic learning-readiness dataset.

## Portfolio purpose

An academic machine-learning example should show the complete workflow—data generation, split strategy, preprocessing, model training, evaluation, and limitations—without using student records.

## Approach

A logistic-regression workflow loads a seeded synthetic dataset, performs a stratified train/test split, standardizes numeric features, trains with NumPy gradient descent, and reports accuracy, precision, recall, F1, ROC AUC, and a confusion matrix.

## Key artifacts

- `synthetic_ml_data.csv` — generated feature table and binary target
- `ml_demo.py` — end-to-end classification workflow
- `metrics.json` — machine-readable metrics created by the script

## How to run or review

```text
python -m pip install -r requirements.txt
python ml_demo.py
```

Run commands from this project directory. All paths inside the code are relative.

## Skills demonstrated

- machine learning
- NumPy
- pandas
- classification
- model evaluation
- reproducibility
- responsible interpretation

## Design decisions

- Use logistic regression for interpretability rather than optimizing for benchmark performance.
- Use a stratified split and report several metrics instead of accuracy alone.
- Keep the target synthetic and frame it as a methodological exercise, not a decision system.

## Limitations

Synthetic relationships are deliberately learnable and do not represent real educational outcomes. The model must not be used to make decisions about actual people.

## Next iteration

Add cross-validation, calibration, threshold analysis, feature-effect plots, and a model card.

## Data statement

Every record, identifier, organization, person, scenario, and result in this project is synthetic.
No employer, client, university, colleague, customer, credential, private path, or sensitive record is used.
