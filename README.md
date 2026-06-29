# Heart Disease Prediction using Machine Learning

This project uses machine learning to predict whether a patient has heart disease, based on their medical test results. The goal is to help hospitals catch high-risk patients early, before serious problems happen.

---

## What This Project Does

1. **Analyze the data** — Look at patient health records and understand patterns.
2. **Build a model** — Train a computer program to predict heart disease from patient data.
3. **Give suggestions to hospitals** — Explain how this model can actually help doctors and patients in real life.

---

## Why This Matters

Heart disease is the world's number one cause of death. Millions of people die every year from it, and many of those deaths could be prevented if the disease was caught early. A machine learning model can act like an early warning system, flagging patients who need closer attention.

---

## About the Data

There are 180 patients in this dataset. For each patient, we have 13 pieces of medical information, plus the final answer of whether they actually had heart disease or not.

| Column | What it means |
|---|---|
| patient_id | A random ID number for each patient (not useful for prediction, so it was removed) |
| slope_of_peak_exercise_st_segment | A heart signal reading taken during exercise |
| thal | Result of a heart blood flow test (normal, fixed defect, or reversible defect) |
| resting_blood_pressure | Blood pressure while resting |
| chest_pain_type | The type of chest pain the patient has (4 types) |
| num_major_vessels | Number of blood vessels visible in a scan (0 to 3) |
| fasting_blood_sugar_gt_120_mg_per_dl | Whether blood sugar is above 120 after fasting (yes/no) |
| resting_ekg_results | Heart electrical activity results while resting |
| serum_cholesterol_mg_per_dl | Cholesterol level in the blood |
| oldpeak_eq_st_depression | A heart signal change measured during exercise |
| sex | 0 means female, 1 means male |
| age | Age of the patient |
| max_heart_rate_achieved | The highest heart rate reached during a test |
| exercise_induced_angina | Whether the patient had chest pain during exercise (yes/no) |
| heart_disease_present | The answer we are trying to predict — 1 means disease present, 0 means no disease |

The data came in two files: one with patient details (values.csv) and one with the final answer (labels.csv). These were combined using the patient ID.

---

## How the Project Was Done

### Step 1: Preparing the Data
- Combined the two data files into one table
- Removed the patient ID column since it has no medical meaning
- Converted the text column (thal) into numbers, since computers can't understand words directly

### Step 2: Exploring the Data
- Checked if any information was missing — none was
- Looked at boxplots to spot unusual or extreme values
- Looked at histograms to see how each value is spread out
- Made a heatmap to see which factors are most connected to heart disease

### Step 3: Preparing for the Model
- Split the data into two parts: one to train the model, one to test it
- Scaled the numbers so that no single feature (like cholesterol, which has big numbers) unfairly overpowers smaller-scale features

### Step 4: Building the Models
Five different prediction methods were tried and compared:
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### Step 5: Checking How Well Each Model Worked
Each model was scored on:
- Accuracy — how many predictions were correct overall
- Precision — when the model said "disease," how often was it actually right
- Recall — out of all patients who actually had disease, how many did the model catch
- F1 Score — a balanced score combining precision and recall

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest | 0.861 | 1.000 | 0.750 | 0.857 |
| SVM | 0.861 | 1.000 | 0.750 | 0.857 |
| Logistic Regression | 0.833 | 0.938 | 0.750 | 0.833 |
| KNN | 0.833 | 0.938 | 0.750 | 0.833 |
| Decision Tree | 0.806 | 0.933 | 0.700 | 0.800 |

Random Forest and SVM scored exactly the same on the basic test. To find which one is actually more reliable, a second check called cross-validation was used. Instead of testing on just one group of patients, it tests the model on several different groups and averages the results — giving a fairer picture.

| Model | Cross-Validation Score |
|---|---|
| Random Forest | 0.79 |
| SVM | 0.77 |

### Best Model: Random Forest

Random Forest scored higher than SVM once tested across multiple groups of patients (79 percent vs 77 percent). This makes it the more dependable choice, even though both models looked equally good in the first test.

---

## Problems Faced and How They Were Solved

1. **Not enough patients (only 180 records)** — With so few patients, results could shift depending on which ones ended up in the test group. Cross-validation was used to get a fairer, more stable comparison.

2. **A text-based column (thal)** — Computers cannot understand words like "normal" or "reversible defect" directly. This was fixed by converting the text into number columns.

3. **Numbers on very different scales** — Cholesterol values are in the hundreds while other values are small decimals. Without fixing this, the model would pay too much attention to big numbers just because they're big. Scaling fixed this.

4. **A few unusually high values (outliers)** — Some patients had very high cholesterol or unusual heart readings. These were kept as-is, since in real medical cases, extreme values are often real warning signs, not mistakes.

5. **Two models scoring the same** — Random Forest and SVM tied on the first test, making it hard to choose. Cross-validation was used to settle this and pick the more reliable one.

---

## Suggestions for the Hospital

1. Use this model as an early warning tool, not as a final diagnosis. It should help doctors decide who needs more testing, not replace a doctor's judgment.

2. The model currently misses about 1 in 4 patients who actually have heart disease. In healthcare, missing a sick patient is far more dangerous than a false alarm, so future versions of this model should focus on reducing these missed cases.

3. Pay close attention to the factors that matter most in the data: number of blood vessels visible in scans, chest pain during exercise, heart signal changes during exercise, and the blood flow test results. These were the strongest signs of heart disease in this dataset.

4. Collect more patient records over time. With only 180 patients, the model's accuracy is limited. More data will make future predictions more trustworthy.

5. Retrain the model regularly as new patient data comes in, so it keeps learning and stays accurate over time.

---

## Tools Used

- Python — programming language
- Pandas and NumPy — for handling and organizing data
- Matplotlib and Seaborn — for creating charts and graphs
- Scikit-learn — for building and testing the machine learning models
- Pickle — for saving the trained model so it can be reused later without retraining

---

## Project Files

- heart-disease-prediction-using-ml-models.ipynb — the main notebook with all the work
- values.csv — patient medical data
- labels.csv — the actual results (disease present or not)
- best_model_rf.pkl — the saved, trained model (Random Forest)
- README.md — this file, explaining the project
- Requirements.txt

---

## How to Run This Project

1. Install the required tools by running this command:
   pip install pandas numpy matplotlib seaborn scikit-learn

2. Make sure values.csv and labels.csv are in the same folder as the notebook.

3. Open the notebook and run every cell from top to bottom. In Jupyter, the safest way is: Kernel menu, then "Restart Kernel and Run All Cells." This avoids any leftover data from previous runs causing wrong results.

4. After running, the trained model and scaler will be saved as best_model_rf.pkl.

---

## How to Use the Saved Model on New Patients

import pickle

with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

scaled_data = scalar.transform(new_patient_data)
prediction = model.predict(scaled_data)

---

## Conclusion

Five different models were built and compared. Random Forest and SVM looked equally good at first, but a more careful comparison using cross-validation showed Random Forest to be slightly more reliable (79 percent vs 77 percent). This model can help hospitals flag at-risk patients early, but it should always be used as a support tool alongside a doctor's own judgment, not as a replacement for one.
