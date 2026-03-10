# 🎓 Student Course Completion Prediction

An end-to-end Supervised ML pipeline that predicts whether a student will complete an online course, trained on 1,00,000 student records and built with Python, Scikit-learn, and deployed via Streamlit.

## 📌 Problem Statement
Online Ed-Tech platforms experience significant student dropouts, resulting in reduced course completion rates and negative impact on the business. 	Hence, there is a need for a predictive model that can accurately estimate a student’s likelihood of completing a course based on learning behavior, engagement and technological factors etc.

## 💡 Suggested Solution
This project aims to predict course completion early so that proactive interventions can be implemented and facilitating the company to ensure student retention and successful completion of the course that proactively increases the company’s profit.

## 📂 Dataset
The dataset used for this project is publicly available on Kaggle:

🔗 **Kaggle Link:**  
https://www.kaggle.com/datasets/nisargpatel344/student-course-completion-prediction-dataset

## 📊 Dataset
- 1,00,000 student records
- 40 features including student demographics, course details, and engagement metrics
- Target variable: Completed (Completed (1) / Not Completed (0))
- Type: Binary Classification
- Dataset includes student academic, course information, app activity information and demographic features.

## 🏗️ ML Pipeline 
Business Problem Understanding -> Data Collection -> Data Cleaning & Preprocessing -> Feature Engineering -> 
 Statistical Test -> Train-Test Split -> Model Training -> Model Comparison -> Threshold Tuning -> Model Evaluation -> Deployment

## 🔍 Approach
- Exploratory Data Analysis (EDA)
- Null value treatment and outlier detection
- Feature encoding (Ordinal, Label, Dummy)
- 14 ML models evaluated including Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, XGBoost with hyper-parameter tuning.
- Best model selected based on train-test consistency

## 🔎 Exploratory Data Analysis (EDA) 
Key observations:
Certain engagement-related features strongly influence completion.
Correlation analysis identified high-impact predictors.
Some features required scaling and missing value treatment.

## 🤖 Models Used 
Logistic Regression (Baseline Model & Best Performing Model),
K-Nearest Neighbour,
Gaussian Naive Bayes,
Decision Tree,
Random Forest,
AdaBoost Classifier,
XGBoost 

## 📈 Evaluation Metrics 
Since this is an imbalanced classification problem, accuracy alone was not sufficient.
The following metrics were used:
Confusion Matrix,
Precision,
Recall,
F1-Score,
ROC-AUC.

## 🧮 Confusion Matrix Interpretation  
True Positives → Correctly predicted completed students;
True Negatives → Correctly predicted non-completed students;
False Positives → Incorrect completion prediction;
False Negatives → Missed at-risk students.

## Best Model: Logistic Regression
The dataset features showed a largely linear relationship with the target variable (Completed). Logistic Regression effectively captured this relationship without overfitting. In an educational platform scenario, interpretability is critical.
- Understanding why a student is predicted as “Not Completed” is often more valuable than slight improvements in accuracy from complex models.
- Therefore, Logistic Regression provides: Reliable predictions, Transparency and also deployment simplicity

## 🏆 Best Model's Scores
- **Logistic Regression**
- Train Accuracy: 60.8%
- Test Accuracy: 60.1%
- Precision: 61%
- Recall: 63%
- F1-score: 62%
- ROC-AUC: 0.65
- No overfitting observed

## 🚀 Deployment
- Built with Streamlit
- Interactive UI with 35 input features
- Real-time prediction with completion probability

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn

## 📌 Business Impact 
By maximizing recall, the model reduces the number of at-risk students that go undetected.
This can help institutions:
Identify dropout risks early
Provide academic counseling
Improve overall completion rate

```

## 📁 Project Structure
```
├── app.py
├── logistic_model.pkl
├── scalers.pkl
├── feature_columns.pkl
├── requirements.txt
├── Student_Course_Completion_Prediction.csv
├── Student_Course_Completion_Prediction.ipynb
└── README.md
```

## Author

**Deepak Arjun K**  
PGP in Data Science & Generative AI | BE Mechanical Engineering












