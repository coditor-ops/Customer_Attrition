# AttritionAI · AI-Powered HR Intelligence Platform

🔗 **Live Demo:** [https://customerattrition.streamlit.app/](https://customerattrition.streamlit.app/)

🧠 **AttritionAI** is an advanced Machine Learning web application built to predict employee attrition risk and provide actionable workforce intelligence. Using the industry-standard **IBM HR Analytics Dataset**, this platform integrates multiple pre-trained machine learning classifiers with a premium, responsive dark glassmorphism dashboard.

---

## 🚀 Key Features

*   **Modern Dark Glassmorphism Interface**: A custom-styled user interface featuring fluid animations, clean typography (Syne & DM Sans), and a glassmorphic aesthetic.
*   **Multi-Model Intelligence**: Compare predictions in real-time across four optimized Machine Learning models:
    *   📐 **Logistic Regression** (Interpretable baseline)
    *   🔷 **Support Vector Machine** (Non-linear boundary classifier)
    *   🔵 **K-Nearest Neighbors** (Instance-based learning)
    *   🚀 **Gradient Boosting** (High-accuracy ensemble tree model)
*   **Structured Profiler**: Dynamic form sections grouping 34 unique employee metrics (demographics, compensation, job details, work history, and work-life balance).
*   **Interactive Analytics & Confidence Gauge**: Dynamic gauge indicator powered by Plotly reflecting prediction risk probability and safety scores.
*   **Explainable AI (Feature Importance)**: Visualizes key risk drivers for supported models to explain exactly why an employee is classified as a flight risk.
*   **Performance Testing Suite**: Local command-line evaluation tool (`evaluate.py`) for generating detailed performance reports on demand.

---

## 📊 Model Performance Summary

Evaluated on the IBM HR Analytics dataset ($N=1,470$ employees):

| Model | Accuracy | Class 1 (Attrition) Precision | Class 1 (Attrition) Recall | Class 1 (Attrition) F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **94.01%** | **0.97** | **0.65** | **0.78** |
| **Support Vector Machine** | **91.43%** | 0.99 | 0.47 | 0.64 |
| **Logistic Regression** | **89.46%** | 0.80 | 0.46 | 0.59 |
| **K-Nearest Neighbors** | **87.14%** | 0.82 | 0.26 | 0.39 |

---

## 🛠️ Technology Stack

*   **Frontend / UI**: [Streamlit](https://streamlit.io/) (with custom CSS injection), Custom HTML5/CSS3
*   **Interactive Visuals**: [Plotly](https://plotly.com/) (Graph Objects & Express)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
*   **Dataset**: IBM HR Analytics Employee Attrition & Performance dataset

---

## 📁 Project Structure

```text
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints
├── Capstone_Classification.ipynb # Initial modeling and training notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset
├── app.py                        # Streamlit web application entry point
├── evaluate.py                   # Script for benchmarking saved models
├── preprocessor.pkl              # Saved sklearn transformation pipeline
├── logistic_model.pkl            # Pre-trained Logistic Regression model
├── svm_model.pkl                 # Pre-trained Support Vector Machine model
├── knn_model.pkl                 # Pre-trained K-Nearest Neighbors model
├── gb_model.pkl                  # Pre-trained Gradient Boosting model
├── requirements.txt              # Project dependencies
└── README.md                     # Repository documentation (this file)
```

---

## ⚙️ Installation & Setup

Follow these steps to run the application and evaluation tools locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-attrition.git
cd customer-attrition
```

### 2. Install Dependencies
Make sure you have python 3.9+ installed. Run:
```bash
pip install -r requirements.txt
```

### 3. Verify Model Performance
To run baseline evaluations and view detailed metrics for all saved models:
```bash
python evaluate.py
```

### 4. Launch the Web App
Run the interactive dashboard:
```bash
streamlit run app.py
```
A new tab will automatically open in your default browser at `http://localhost:8501`.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have ideas for model improvements, additional features, or UI adjustments.
