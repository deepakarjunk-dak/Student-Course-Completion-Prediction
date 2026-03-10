import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load all saved files
model = pickle.load(open("logistic_model.pkl", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.set_page_config(page_title="Student Course Completion Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Course Completion Predictor")
st.markdown("Fill in the student details below to predict whether they will complete the course.")
st.divider()

# --- INPUT FORM ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Student Info")
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Age = st.number_input("Age", min_value=15, max_value=60, value=22)
    Education_Level = st.selectbox("Education Level", ["HighSchool", "Diploma", "Bachelor", "Master", "PhD"])
    Employment_Status = st.selectbox("Employment Status", ["Student", "Employed", "Self-Employed", "Unemployed"])
    City = st.selectbox("City", ["Ahmedabad", "Bengaluru", "Bhopal", "Chennai", "Delhi", "Hyderabad",
                                  "Indore", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Nagpur",
                                  "Pune", "Surat", "Vadodara"])
    Device_Type = st.selectbox("Device Type", ["Laptop", "Mobile", "Tablet"])
    Internet_Connection_Quality = st.selectbox("Internet Quality", ["Low", "Medium", "High"])
    Fee_Paid = st.selectbox("Fee Paid", ["No", "Yes"])
    Discount_Used = st.selectbox("Discount Used", ["No", "Yes"])
    Payment_Mode = st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "Free", "NetBanking", "Scholarship", "UPI"])
    Payment_Amount = st.number_input("Payment Amount (₹)", min_value=0, max_value=100000, value=5000)

with col2:
    st.subheader("📚 Course Info")
    Course_Name = st.selectbox("Course Name", ["Data Analysis with Python", "Digital Marketing Essentials",
                                                "Excel for Business", "Introduction to AI",
                                                "Machine Learning A-Z", "Python Basics",
                                                "Statistics for Data Science", "UI/UX Design Fundamentals"])
    Category = st.selectbox("Category", ["Business", "Design", "Marketing", "Math", "Programming"])
    Course_Level = st.selectbox("Course Level", ["Beginner", "Intermediate", "Advanced"])
    Course_Duration_Days = st.number_input("Course Duration (Days)", min_value=1, max_value=365, value=60)
    Instructor_Rating = st.slider("Instructor Rating", 1.0, 5.0, 4.0, 0.1)

with col3:
    st.subheader("📊 Engagement Metrics")
    Login_Frequency = st.number_input("Login Frequency", min_value=0, max_value=100, value=10)
    Average_Session_Duration_Min = st.number_input("Avg Session Duration (Min)", min_value=0, max_value=300, value=45)
    Video_Completion_Rate = st.slider("Video Completion Rate", 0.0, 1.0, 0.6, 0.01)
    Discussion_Participation = st.number_input("Discussion Participation", min_value=0, max_value=100, value=5)
    Time_Spent_Hours = st.number_input("Time Spent (Hours)", min_value=0, max_value=500, value=50)
    Days_Since_Last_Login = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=7)
    Notifications_Checked = st.number_input("Notifications Checked", min_value=0, max_value=500, value=20)
    Peer_Interaction_Score = st.slider("Peer Interaction Score", 0.0, 10.0, 5.0, 0.1)
    Assignments_Submitted = st.number_input("Assignments Submitted", min_value=0, max_value=50, value=8)
    Assignments_Missed = st.number_input("Assignments Missed", min_value=0, max_value=50, value=2)
    Quiz_Attempts = st.number_input("Quiz Attempts", min_value=0, max_value=50, value=5)
    Quiz_Score_Avg = st.slider("Quiz Score Avg", 0.0, 100.0, 60.0, 0.1)
    Project_Grade = st.slider("Project Grade", 0.0, 100.0, 65.0, 0.1)
    Progress_Percentage = st.slider("Progress Percentage", 0.0, 100.0, 50.0, 0.1)
    Rewatch_Count = st.number_input("Rewatch Count", min_value=0, max_value=100, value=3)
    App_Usage_Percentage = st.slider("App Usage %", 0.0, 100.0, 50.0, 0.1)
    Reminder_Emails_Clicked = st.number_input("Reminder Emails Clicked", min_value=0, max_value=100, value=5)
    Support_Tickets_Raised = st.number_input("Support Tickets Raised", min_value=0, max_value=20, value=1)
    Satisfaction_Rating = st.slider("Satisfaction Rating", 1.0, 5.0, 3.5, 0.1)

# --- ENCODING MAPS ---
education_map = {"PhD": 0.0, "Master": 1.0, "Bachelor": 2.0, "Diploma": 3.0, "HighSchool": 4.0}
internet_map = {"High": 0.0, "Medium": 1.0, "Low": 2.0}
course_level_map = {"Advanced": 0.0, "Intermediate": 1.0, "Beginner": 2.0}

gender_classes = sorted(["Male", "Female", "Other"])
employment_classes = sorted(["Student", "Employed", "Self-Employed", "Unemployed"])
city_classes = sorted(["Ahmedabad", "Bengaluru", "Bhopal", "Chennai", "Delhi", "Hyderabad",
                        "Indore", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Nagpur",
                        "Pune", "Surat", "Vadodara"])
device_classes = sorted(["Laptop", "Mobile", "Tablet"])
course_classes = sorted(["Data Analysis with Python", "Digital Marketing Essentials",
                          "Excel for Business", "Introduction to AI",
                          "Machine Learning A-Z", "Python Basics",
                          "Statistics for Data Science", "UI/UX Design Fundamentals"])
category_classes = sorted(["Business", "Design", "Marketing", "Math", "Programming"])
payment_classes = sorted(["Credit Card", "Debit Card", "Free", "NetBanking", "Scholarship", "UPI"])

# --- PREDICT BUTTON ---
st.divider()
if st.button("🔍 Predict Course Completion", type="primary", use_container_width=True):

    input_dict = {
        "Gender": gender_classes.index(Gender),
        "Age": Age,
        "Education_Level": education_map[Education_Level],
        "Employment_Status": employment_classes.index(Employment_Status),
        "City": city_classes.index(City),
        "Device_Type": device_classes.index(Device_Type),
        "Internet_Connection_Quality": internet_map[Internet_Connection_Quality],
        "Course_Name": course_classes.index(Course_Name),
        "Category": category_classes.index(Category),
        "Course_Level": course_level_map[Course_Level],
        "Course_Duration_Days": Course_Duration_Days,
        "Instructor_Rating": Instructor_Rating,
        "Login_Frequency": Login_Frequency,
        "Average_Session_Duration_Min": Average_Session_Duration_Min,
        "Video_Completion_Rate": Video_Completion_Rate,
        "Discussion_Participation": Discussion_Participation,
        "Time_Spent_Hours": Time_Spent_Hours,
        "Days_Since_Last_Login": Days_Since_Last_Login,
        "Notifications_Checked": Notifications_Checked,
        "Peer_Interaction_Score": Peer_Interaction_Score,
        "Assignments_Submitted": Assignments_Submitted,
        "Assignments_Missed": Assignments_Missed,
        "Quiz_Attempts": Quiz_Attempts,
        "Quiz_Score_Avg": Quiz_Score_Avg,
        "Project_Grade": Project_Grade,
        "Progress_Percentage": Progress_Percentage,
        "Rewatch_Count": Rewatch_Count,
        "Payment_Mode": payment_classes.index(Payment_Mode),
        "Fee_Paid": 1 if Fee_Paid == "Yes" else 0,
        "Discount_Used": 1 if Discount_Used == "Yes" else 0,
        "Payment_Amount": Payment_Amount,
        "App_Usage_Percentage": App_Usage_Percentage,
        "Reminder_Emails_Clicked": Reminder_Emails_Clicked,
        "Support_Tickets_Raised": Support_Tickets_Raised,
        "Satisfaction_Rating": Satisfaction_Rating,
    }

    # Build dataframe in correct feature order
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Apply RobustScaler to numerical columns only
    for col in scalers.keys():
        if col in input_df.columns:
            input_df[col] = scalers[col].transform(input_df[[col]].values.reshape(-1, 1))

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # --- RESULT ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.success("✅ This student is **LIKELY TO COMPLETE** the course!")
        else:
            st.error("❌ This student is **AT RISK of NOT completing** the course!")
    with res_col2:
        st.metric("Completion Probability", f"{probability:.1%}")

    st.caption("Model: Logistic Regression | Best among 14 models evaluated | Dataset: 1,00,000 students | Accuracy: 60.1%")
