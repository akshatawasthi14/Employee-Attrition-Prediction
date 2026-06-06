import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load files
model = joblib.load("attrition_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
threshold = joblib.load("threshold.pkl")

st.title("Employee Attrition Prediction")
st.markdown("""
This application predicts the likelihood of employee attrition
using a Logistic Regression model trained on employee demographics,
job characteristics, compensation, work-life balance, and career history.

**Note:** This prediction is intended for educational and analytical
purposes only and should not be used as the sole basis for HR decisions.
""")
st.header("Employee Information")

# -------------------------
# Numerical Inputs
# -------------------------

age = st.number_input("Age", 18, 65, 30)

distance_from_home = st.number_input(
    "Distance From Home",
    0,
    50,
    5
)

education = st.selectbox(
    "Education Level",
    [1, 2, 3, 4, 5]
)

job_involvement = st.selectbox(
    "Job Involvement",
    [1, 2, 3, 4]
)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    100000,
    10000
)

num_companies_worked = st.number_input(
    "Number of Companies Worked",
    0,
    20,
    2
)

performance_rating = st.selectbox(
    "Performance Rating",
    [1, 2, 3, 4]
)

stock_option_level = st.selectbox(
    "Stock Option Level",
    [0, 1, 2, 3]
)

training_times_last_year = st.number_input(
    "Training Times Last Year",
    0,
    10,
    2
)

work_life_balance = st.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4]
)

years_in_current_role = st.number_input(
    "Years In Current Role",
    0,
    40,
    3
)

# -------------------------
# Categorical Inputs
# -------------------------

business_travel = st.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)

department = st.selectbox(
    "Department",
    [
        "Sales",
        "Research & Development",
        "Human Resources"
    ]
)

education_field = st.selectbox(
    "Education Field",
    [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Human Resources",
        "Other"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

overtime = st.selectbox(
    "OverTime",
    [
        "Yes",
        "No"
    ]
)

# -------------------------
# Engineered Feature Inputs
# -------------------------
st.info("""
The following values are used to calculate internal career metrics
such as tenure ratio, promotion stagnation, manager stability,
and satisfaction score.
""")
st.header("Career History")

total_working_years = st.number_input(
    "Total Working Years",
    0,
    50,
    10
)

years_at_company = st.number_input(
    "Years At Company",
    0,
    40,
    5
)

years_since_last_promotion = st.number_input(
    "Years Since Last Promotion",
    0,
    20,
    2
)

years_with_curr_manager = st.number_input(
    "Years With Current Manager",
    0,
    20,
    3
)

job_satisfaction = st.selectbox(
    "Job Satisfaction (1-4)",
    [1, 2, 3, 4]
)

environment_satisfaction = st.selectbox(
    "Environment Satisfaction (1-4)",
    [1, 2, 3, 4]
)

relationship_satisfaction = st.selectbox(
    "Relationship Satisfaction (1-4)",
    [1, 2, 3, 4]
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Attrition"):

    tenure_ratio = years_at_company / (total_working_years + 1)

    income_per_year_exp = (
        monthly_income /
        (total_working_years + 1)
    )

    satisfaction_score = (
        job_satisfaction +
        environment_satisfaction +
        relationship_satisfaction
    ) / 3

    promotion_stagnation = (
        years_since_last_promotion /
        (years_at_company + 1)
    )

    manager_stability = (
        years_with_curr_manager /
        (years_at_company + 1)
    )

    input_df = pd.DataFrame({

        "Age": [age],
        "BusinessTravel": [business_travel],
        "Department": [department],
        "DistanceFromHome": [distance_from_home],
        "Education": [education],
        "EducationField": [education_field],
        "Gender": [gender],
        "JobInvolvement": [job_involvement],
        "JobRole": [job_role],
        "MaritalStatus": [marital_status],
        "MonthlyIncome": [monthly_income],
        "NumCompaniesWorked": [num_companies_worked],
        "OverTime": [overtime],
        "PerformanceRating": [performance_rating],
        "StockOptionLevel": [stock_option_level],
        "TrainingTimesLastYear": [training_times_last_year],
        "WorkLifeBalance": [work_life_balance],
        "YearsInCurrentRole": [years_in_current_role],
        "TenureRatio": [tenure_ratio],
        "IncomePerYearExp": [income_per_year_exp],
        "SatisfactionScore": [satisfaction_score],
        "PromotionStagnation": [promotion_stagnation],
        "ManagerStability": [manager_stability]
    })

    # Apply SAME log transforms used during training

    log_cols = [
        'MonthlyIncome',
        'IncomePerYearExp',
        'NumCompaniesWorked',
        'DistanceFromHome',
        'YearsInCurrentRole',
        'PromotionStagnation'
    ]

    for col in log_cols:
        input_df[col] = np.log1p(input_df[col])


    transformed = preprocessor.transform(input_df)

    proba = model.predict_proba(transformed)

    attrition_probability = float(proba[0][1])

    prediction = int(attrition_probability >= threshold)

    st.metric(
        "Estimated Attrition Risk",
        f"{attrition_probability:.2%}"
    )

    st.caption(
        f"Decision Threshold Used: {threshold:.2f}"
    )

    if prediction == 1:
        st.error(
            "⚠️ Employee is likely to leave the organization."
        )
    else:
        st.success(
            "✅ Employee is likely to stay with the organization."
    )