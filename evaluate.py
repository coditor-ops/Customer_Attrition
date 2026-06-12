import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
y = df['Attrition'].map({'Yes': 1, 'No': 0})

FEATURE_NAMES = [
    "Age","BusinessTravel","DailyRate","Department","DistanceFromHome",
    "Education","EducationField","EmployeeCount","EmployeeNumber",
    "EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement",
    "JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome",
    "MonthlyRate","NumCompaniesWorked","Over18","OverTime","PercentSalaryHike",
    "PerformanceRating","RelationshipSatisfaction","StandardHours",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
    "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
    "YearsSinceLastPromotion","YearsWithCurrManager"
]

X_raw = df[FEATURE_NAMES]
preprocessor = joblib.load("preprocessor.pkl")
X = preprocessor.transform(X_raw)

models = {
    "Logistic Regression": "logistic_model.pkl",
    "Support Vector Machine": "svm_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
}

for name, file in models.items():
    try:
        model = joblib.load(file)
        y_pred = model.predict(X)
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(classification_report(y, y_pred, target_names=["Retained (0)", "Attrition (1)"]))
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")
