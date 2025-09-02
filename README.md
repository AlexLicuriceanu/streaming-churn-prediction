# Movie streaming platform churn predictor based on historical data.

## Premade Datasets - all have the same distributions, only the number of samples is different
- Small - 100.000 samples (5.61 MB)
- Medium - 500.000 samples (28.0 MB)
- Large - 1.000.000 samples (56.1 MB)
- Very Large - 5.000.000 samples (280.0 MB)

## Setup
```bash
pip3 install -r requirements.txt
```

## Dataset generation 
- In `generate-dataset.py`, variable `N_USERS` specifies how many samples to generate, `FILE_NAME` specifies the CSV file's name.
- Base probabilities and adjusting factors can be modified inside each attribute's generator function.
```bash
python3 generate-dataset.py
```
