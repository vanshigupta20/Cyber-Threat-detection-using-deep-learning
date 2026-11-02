# Cyber Threat Detection Using Deep Learning
### A Network Intrusion Detection System (NIDS) on CIC-IDS2017

> This is a machine-learning-based academic prototype trained and evaluated on the CIC-IDS2017 dataset.
> It is not a real-time enterprise IDS and it cannot detect every cyber attack.

The system classifies network flows (CICFlowMeter features) as **Normal (0)** or **Attack (1)** using an
Artificial Neural Network (TensorFlow/Keras), and has a Streamlit dashboard for CSV-based intrusion detection.

```
CIC-IDS2017 CSVs -> cleaning -> binary labels -> train/val/test split -> scaler (fit on train only)
-> ANN (class weights + early stopping) -> evaluation on unseen test set -> saved model
-> Streamlit dashboard -> CSV intrusion prediction -> downloadable results
```

## Project structure
```
cyber_threat_detection/
├── app.py                      Streamlit dashboard
├── requirements.txt
├── README.md
├── .streamlit/config.toml      dark SOC theme, 500 MB upload limit
├── dataset/
│   ├── raw/                    put the CIC-IDS2017 CSV files here
│   └── processed/              created by preprocess.py
├── models/                     created by preprocess.py / train_model.py
├── outputs/
│   ├── plots/                  PNG charts for the report
│   ├── metrics/                JSON/TXT metrics read by the dashboard
│   └── predictions/            CSV results from src/predict.py
└── src/
    ├── __init__.py
    ├── config.py               shared paths and constants (threshold, split sizes ...)
    ├── data_utils.py           cleaning functions shared by training and prediction
    ├── preprocess.py           step 1
    ├── train_model.py          step 2
    ├── evaluate_model.py       step 3
    └── predict.py              prediction engine (used by app.py, also a CLI)
```

## Dataset
1. Open https://www.unb.ca/cic/datasets/ids-2017.html and use the download link (a short form).
2. In the CIC-IDS-2017 download folder, open `CSVs/` and download **`MachineLearningCSV.zip`**
   (not the PCAPs, not `GeneratedLabelledFlows.zip`).
3. Extract it and put the 8 CSV files in `dataset/raw/` (a sub-folder such as
   `dataset/raw/MachineLearningCVE/` also works; CSVs are found recursively):
   ```
   Monday-WorkingHours.pcap_ISCX.csv
   Tuesday-WorkingHours.pcap_ISCX.csv
   Wednesday-workingHours.pcap_ISCX.csv
   Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   Friday-WorkingHours-Morning.pcap_ISCX.csv
   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

## Setup and run
```bash
conda create -n cyberids python=3.11 -y
conda activate cyberids
pip install -r requirements.txt

python src/preprocess.py --mode final     # or: --mode dev  (10 % sample, quick test)
python src/train_model.py
python src/evaluate_model.py
streamlit run app.py
```
Optional terminal prediction:
```bash
python src/predict.py --input dataset/processed/sample_test_flows.csv
```

## DEVELOPMENT MODE vs FINAL TRAINING MODE
| | `--mode dev` | `--mode final` (default) |
|---|---|---|
| Data used | random sample of each CSV (`--sample-fraction`, default 0.10) | every row |
| Purpose | checking that the pipeline works | results for the report |
| Dashboard | shows a DEVELOPMENT MODE warning | normal |

The mode is saved in `outputs/metrics/dataset_summary.json`, so a sample-based model is never
silently presented as the final result.

## Preprocessing steps (`src/preprocess.py`)
1. Read each CSV **one at a time** (latin-1 encoding, because web-attack labels contain byte `0x96`).
2. Strip spaces from column names (`" Destination Port"` -> `"Destination Port"`).
3. Remove empty rows (the Thursday web-attack file ends with many empty rows).
4. Remove identifier columns (`Destination Port`, the duplicate `Fwd Header Length.1`, and IP/port/timestamp
   columns if present) and any non-numeric (categorical/text) columns.
5. Convert features to `float32`; turn `+Infinity`/`-Infinity` into NaN; drop rows with NaN.
6. Remove duplicate records (inside each file, then across files).
7. Labels: `BENIGN -> 0`, every other label -> `1`.
8. Stratified split: **70 % train, 10 % validation, 20 % test**.
9. Drop features that are constant **in the training split**.
10. Fit `StandardScaler` **on the training split only**, then transform train, validation and test.
11. Save arrays (`.npy`), `scaler.pkl`, `feature_columns.pkl`, a dataset summary JSON, and plots.

## Avoiding data leakage
```
Raw dataset -> Clean -> Train/Val/Test split -> Fit scaler ONLY on training data
-> Transform training data -> Transform val/test data with the SAME scaler -> Train -> Evaluate on unseen test data
```
If the scaler were fitted on the full dataset, the mean and standard deviation of the test rows would influence
the training inputs, so the test score would look better than it really is. The same rule applies to feature
selection (constant columns are decided on training rows only) and to class imbalance handling
(class weights come from training labels only; nothing is oversampled and the test set is never changed).
De-duplication is done before splitting so identical flows cannot appear in both train and test.

## Model (`src/train_model.py`)
```
Input(n features) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)
Optimizer: Adam (lr 0.001)   Loss: binary cross-entropy   Metric: accuracy
Class weights: 'balanced' (from training labels)   EarlyStopping(val_loss, patience 3, best weights restored)
```

## Evaluation (`src/evaluate_model.py`)
Accuracy, precision, recall, F1, confusion matrix (TN, FP, FN, TP), `classification_report`, and the
detection rate per original attack type, all on the unseen test set. The dashboard reads these values from
`outputs/metrics/evaluation_metrics.json`; no performance number is hard-coded anywhere.

```
                 Predicted
              Normal   Attack
Actual Normal   TN       FP
Actual Attack   FN       TP
```
A **false negative** is an attack classified as normal. It is the most dangerous error for an IDS because
no alert is raised.

## Consistency between files
All scripts import `src/config.py` (paths, label encoding, threshold 0.5) and `src/data_utils.py`
(the same cleaning functions for training and prediction). `predict.py` re-orders uploaded columns to the order
stored in `feature_columns.pkl` and uses the saved `scaler.pkl`. If `preprocess.py` is run again after training,
the app and `evaluate_model.py` detect that the model no longer matches the scaler and ask you to retrain.

## Threat level
`LOW < 0.30 <= MEDIUM < 0.70 <= HIGH`, based on attack probability. This is a **project visualisation rule,
not an official cybersecurity severity standard**. The Normal/Attack decision always uses the 0.5 threshold.

## Limitations
- Trained on one 2017 lab dataset; performance on other networks will be lower.
- Needs CICFlowMeter flow features; it does not capture live packets.
- Binary classification only (the attack type is not predicted).
