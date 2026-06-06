# Employee Attrition Prediction

A machine learning project that predicts employee attrition using HR analytics data from the IBM HR Analytics Employee Attrition & Performance dataset.

The objective is to identify employees who are at risk of leaving the organization by leveraging demographic information, compensation data, job-related attributes, satisfaction metrics, and engineered behavioral features.

---

## Dataset

Dataset: IBM HR Analytics Employee Attrition & Performance

- Rows: 1470
- Original Features: 35
- Target Variable: Attrition

Target Encoding:

| Attrition | Value |
|------------|--------|
| No | 0 |
| Yes | 1 |

The dataset is imbalanced, with only 16.1% attrition cases.

---

## Application Preview

### Prediction Dashboard

![Dashboard](screenshots/app-interface.png)

### Example Prediction

![Prediction](screenshots/low-risk-prediction-example.png)

---

## Project Workflow

### Data Cleaning

The following columns were removed because they either contained constant values or acted as unique identifiers with no predictive value:

- EmployeeCount
- StandardHours
- Over18
- EmployeeNumber

The target variable was encoded as:

```python
Yes -> 1
No  -> 0
```

---

### Exploratory Data Analysis

Several analyses were performed to understand attrition patterns across both categorical and numerical variables.

Key observations:

- Employees working overtime showed significantly higher attrition rates.
- Single employees had higher attrition than married employees.
- Sales Representatives exhibited the highest attrition among job roles.
- Frequent business travel correlated with increased employee turnover.
- Lower monthly income was associated with higher attrition.
- Employees with shorter tenure were more likely to leave.
- Lower satisfaction metrics corresponded with increased attrition.

EDA included:

- Class imbalance analysis
- Categorical attrition rate analysis
- Numerical feature distribution analysis
- Correlation heatmaps
- Outlier inspection using boxplots
- Job Level vs Department attrition heatmaps

---

### Feature Engineering

To capture employee behavior more effectively, several domain-inspired features were created.

#### TenureRatio

Represents the proportion of an employee's career spent at the current company.

#### IncomePerYearExp

Normalizes monthly income by total work experience.

#### SatisfactionScore

Average of:

- JobSatisfaction
- EnvironmentSatisfaction
- RelationshipSatisfaction

#### PromotionStagnation

Measures promotion frequency relative to company tenure.

#### ManagerStability

Measures consistency of managerial supervision.

---

### Feature Reduction

After creating engineered features, the original source variables were removed to reduce redundancy.

Final Dataset:

- 23 Features
- 1470 Rows

---

### Skewness Treatment

Feature skewness was analyzed and highly skewed numerical variables were transformed using:

```python
np.log1p()
```

Applied to:

- MonthlyIncome
- IncomePerYearExp
- NumCompaniesWorked
- DistanceFromHome
- YearsInCurrentRole
- PromotionStagnation

This reduced skewness and improved model stability.

---

### Preprocessing Pipeline

A Scikit-Learn ColumnTransformer pipeline was used.

Numerical Features:

- RobustScaler

Categorical Features:

- OneHotEncoder

---

### Train-Test Split

Training Samples: 1176
Testing Samples: 294

---

### Class Imbalance Handling

Because attrition represented only 16.1% of observations, SMOTEENN was evaluated.

SMOTEENN combines:

- SMOTE (Synthetic Minority Oversampling)
- Edited Nearest Neighbours (undersampling)

The technique successfully balanced the training dataset and was included in experimentation.

---

### Model Benchmarking

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machine
- K-Nearest Neighbours
- Naive Bayes
- XGBoost

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

### Hyperparameter Tuning

Hyperparameter optimization was performed using:

```python
GridSearchCV
```

and

```python
RandomizedSearchCV
```

Cross-validation was conducted using Stratified K-Fold splitting.

Optimization metric:

```python
ROC-AUC
```

---

### Final Model Selection

Although multiple ensemble methods and resampling approaches were evaluated, the tuned Logistic Regression model trained on the original dataset proved to be the best one.

---

### Threshold Optimization

Instead of relying on the default classification threshold of 0.50, threshold tuning was performed to maximize F1 Score.

Optimal Threshold:

```python
0.25
```

Using a lower threshold improved the balance between precision and recall for the minority attrition class.

---

## Final Model

After model benchmarking, hyperparameter tuning, and threshold optimization, Logistic Regression was selected as the final model due to its strong balance of predictive performance, interpretability, and deployment simplicity.

While several ensemble methods achieved competitive results, Logistic Regression delivered the highest ROC-AUC score and responded well to threshold optimization, making it the most suitable choice for deployment.

### Final Performance (Optimized Threshold = 0.25)

| Metric | Score |
|----------|----------|
| Accuracy | 85.03% |
| Precision | 52.63% |
| Recall | 63.83% |
| F1 Score | 57.69% |
| ROC-AUC | 0.801 |

### Why Threshold Optimization?

The default Logistic Regression classification threshold of **0.50** produced higher precision but missed a substantial number of attrition cases.

To improve detection of employees likely to leave, thresholds between **0.10 and 0.89** were evaluated using F1 Score.

The optimal threshold was found to be:

```python
0.25
```

This increased the Recall and F1 score at the cost of a modest reduction in precision.

For an employee attrition use case, improving recall is desirable because identifying potential attrition risks is generally more valuable than maximizing precision alone.

### Classification Report

| Class | Precision | Recall | F1 Score |
|---------|---------|---------|---------|
| Stay (0) | 0.93 | 0.89 | 0.91 |
| Leave (1) | 0.53 | 0.64 | 0.58 |

---

## Deployment

The trained artifacts were exported using Joblib:

```text
attrition_model.pkl
preprocessor.pkl
threshold.pkl
```

A Streamlit application was developed to:

- Accept employee information
- Recreate engineered features
- Apply the same preprocessing pipeline used during training
- Generate attrition probabilities
- Classify employees using the optimized threshold

During deployment, an inference mismatch caused incorrect predictions. The issue was traced to missing log transformations in the Streamlit pipeline and was resolved by applying the same feature transformations used during model training.

---

## Repository Structure

```
employee_attrition_prediction/
│
├── EmployeeAttritionPredictionModel.ipynb
├── app.py
├── attrition_model.pkl
├── preprocessor.pkl
├── threshold.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- XGBoost
- Joblib
- Streamlit

---

## Running Locally

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```