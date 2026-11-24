# 🩺 Diabetes Prediction System

A Machine Learning project that predicts whether a person falls into the diabetes or non-diabetes category based on health-related and lifestyle information.

The project uses a **Random Forest Classifier** trained on a diabetes health indicators dataset and provides an interactive web interface built with **Streamlit**.

## 📌 Project Overview

Diabetes is a chronic health condition that affects how the body regulates blood glucose.

This project demonstrates how Machine Learning can be used to analyze health indicators and classify patient records into two categories:

- **0 – No Diabetes**
- **1 – Diabetes**

Users can enter patient information through a simple web interface and receive a prediction from the trained machine learning model.

> **Disclaimer:** This project is developed for educational purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## ✨ Features

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Random Forest classification
- Model evaluation
- Model serialization using Joblib
- Interactive Streamlit web application
- User-friendly patient information form
- Diabetes prediction with class probability estimates

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model saving and loading |
| Streamlit | Web application |

## 📂 Project Structure

```text
Diabetes_Prediction/
│
├── dataset/
│   └── diabetes.csv
│
├── models/
│   └── diabetes_model.pkl
│
├── reports/
│   ├── target_distribution.png
│   ├── bmi_distribution.png
│   ├── age_distribution.png
│   └── correlation_heatmap.png
│
├── train.py
├── app.py
└── README.md
```

### File Description

- `dataset/diabetes.csv` – Dataset used for training.
- `models/diabetes_model.pkl` – Saved trained machine learning model.
- `reports/` – Generated data visualization reports.
- `train.py` – Performs data analysis, visualization, model training, evaluation, and model saving.
- `app.py` – Runs the interactive Streamlit application.
- `README.md` – Project documentation.

## 📊 Dataset Information

The dataset contains **70,692 records and 22 columns**, including:

- 21 input features
- 1 target variable (`Diabetes_binary`)

### Target Variable

| Value | Meaning |
|---|---|
| 0 | No Diabetes |
| 1 | Diabetes |

### Input Features

| Feature | Description |
|---|---|
| HighBP | High blood pressure |
| HighChol | High cholesterol |
| CholCheck | Cholesterol check |
| BMI | Body Mass Index |
| Smoker | Smoking history |
| Stroke | History of stroke |
| HeartDiseaseorAttack | Heart disease or heart attack history |
| PhysActivity | Physical activity |
| Fruits | Fruit consumption |
| Veggies | Vegetable consumption |
| HvyAlcoholConsump | Heavy alcohol consumption |
| AnyHealthcare | Healthcare coverage |
| NoDocbcCost | Unable to visit doctor due to cost |
| GenHlth | General health condition |
| MentHlth | Number of days mental health was not good |
| PhysHlth | Number of days physical health was not good |
| DiffWalk | Difficulty walking |
| Sex | Sex category |
| Age | Age category |
| Education | Education level |
| Income | Household income category |

### Important Encoding Information

Some dataset features are stored as numerical categories.

| Feature | Encoding |
|---|---|
| Sex | 0 = Female, 1 = Male |
| Age | 1–13, representing age groups from 18–24 to 80+ |
| Education | 1–6, representing education levels |
| Income | 1–8, representing annual household income brackets |

The Streamlit application displays readable labels and converts the selections into the numerical encodings expected by the trained model.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate into the project directory:

```bash
cd Diabetes_Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## 🚀 How to Run the Project

### Step 1: Train the Machine Learning Model

Run:

```bash
python train.py
```

This will:

1. Load the dataset.
2. Display dataset information.
3. Perform exploratory data analysis.
4. Generate visualizations.
5. Split the dataset into training and testing sets.
6. Train the Random Forest Classifier.
7. Evaluate model performance.
8. Save the trained model inside the `models/` directory.

### Step 2: Launch the Web Application

After training is complete, execute:

```bash
streamlit run app.py
```

The application will open in your browser.

Default local address:

```text
http://localhost:8501
```

**Important:** Run the application using `streamlit run app.py`, not `python app.py`.

## 🧠 Machine Learning Model

This project uses the **Random Forest Classifier**.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

### Model Training

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The model uses the following configuration:

- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Maximum depth: 15
- Random state: 42

### Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Actual evaluation results are generated when `train.py` is executed.

## 🌐 Streamlit Application

The application provides an interactive interface with four main sections:

1. **Personal Information** – Age group, sex, education, income, and BMI.
2. **Medical History** – Blood pressure, cholesterol, stroke, heart disease, and healthcare information.
3. **Lifestyle Information** – Smoking, physical activity, fruit consumption, vegetable consumption, and alcohol consumption.
4. **Health Assessment** – General health, mental health days, and physical health days.

After entering the information, users can generate a prediction and view the model's estimated probabilities for both classes.

## ⚠️ Limitations

- The model's predictions depend on the quality and distribution of the training dataset.
- The dataset uses categorical age and income groups rather than exact values.
- The dataset represents survey-based health indicators and may not represent every population.
- Model probabilities are not equivalent to clinically validated individual diabetes risk.
- The application is intended for educational and demonstration purposes only.

## 🔮 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Testing additional classification algorithms
- Feature importance analysis
- Cross-validation
- Model comparison
- Improved application interface
- Model explainability using SHAP
- Deployment to a cloud platform

## 👨‍💻 Author

**Your Name**

Machine Learning Project – Diabetes Prediction System

---

*Built using Python, Scikit-learn, and Streamlit.*