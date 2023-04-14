import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
# header_images = Image.open('assets/header_images.jpg')
# st.image(header_images)

# Add some information about the service
st.title("Telemarketing Subscribe Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "telemarketing_form"):
    # Create select box input
    # Create box for number input
    age = st.number_input(
        label = "\tEnter Age Value :",
        min_value = 17,
        max_value = 100,
        help = "Value range from 15 to 100"
    )
    
    job = st.selectbox(
        label = "\tJob :",
        options = (
            "admin.",
            "blue-collar",
            "technician",
            "services",
            "management",
            "retired",
            "entrepreneur",
            "self-employed",
            "housemaid",
            "unemployed",
            "student",
            "unknown",
        )
    )
    
    marital = st.selectbox(
        label = "\tMarital Status :",
        options = (
            "married",
            "single",
            "divorced",
            "unknown",
        )
    )

    education = st.selectbox(
        label = "\tEducation :",
        options = (
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "university.degree",
            "professional.course",
            "unknown",
            "illiterate",
        )
    )
    
    default = st.selectbox(
        label = "\tHas Credit in Default?",
        options = (
            "no",
            "yes",
            "unknown",
        )
    )
    
    housing = st.selectbox(
        label = "\tHas Housing Loan?",
        options = (
            "no",
            "yes",
            "unknown",
        )
    )
    
    loan = st.selectbox(
        label = "\tHas Personal Loan?",
        options = (
            "no",
            "yes",
            "unknown",
        )
    )
    
    contact = st.selectbox(
        label = "\tContact (contact communication type) :",
        options = (
            "cellular",
            "telephone",
        )
    )
    
    month = st.selectbox(
        label = "\t Last contact month of year :",
        options = (
            "may",
            "jul",
            "aug",
            "jun",
            "nov",
            "apr",
            "oct",
            "sep",
            "mar",
            "dec",
        )
    )
    
    day_of_week = st.selectbox(
        label = "\t Last contact day of the week :",
        options = (
            "thu",
            "mon",
            "wed",
            "tue",
            "fri",
        )
    )
    
    duration = st.number_input(
        label = "\tLast contact duration :",
        min_value = 0,
        max_value = 4918,
        help = "Value range from 0 to 4918"
    )
    
    campaign = st.number_input(
        label = "\t Number of contacts performed during this campaign and for this client :",
        min_value = 1,
        max_value = 56,
        help = "Value range from 0 to 4918"
    )
    
    pdays = st.number_input(
        label = "\t number of days that passed by after the client was last contacted from a previous campaign (999 means client was not previously contacted) :",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )
    
    previous = st.number_input(
        label = "\t Number of contacts performed before this campaign and for this client :",
        min_value = 0,
        max_value = 7,
        help = "Value range from 0 to 7"
    )
    
    poutcome = st.selectbox(
        label = "\t Outcome of the previous marketing campaign :",
        options = (
            "nonexistent",
            "failure",
            "success",
        )
    )
    
    empvarrate = st.number_input(
        label="Employment variation rate - quarterly indicator",
        min_value = -3.4,
        max_value = 1.4,
        format="%.2f")
    
    conspriceidx = st.number_input(
        label="Consumer price index - monthly indicator",
        min_value = 90.0,
        max_value = 95.0,
        format="%.2f")
    
    consconfidx = st.number_input(
        label="Consumer confidence index - monthly indicator",
        min_value = -51.0,
        max_value = -26.0,
        format="%.2f")
    
    euribor3m = st.number_input(
        label="Euribor 3 month rate - daily indicator",
        min_value = 0.63,
        max_value = 5.1,
        format="%.2f")
    
    nremployed = st.number_input(
        label="Number of employees - quarterly indicator",
        min_value = 4963.0,
        max_value = 5229.0,
        format="%.2f")
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "age" : age,
            "duration" : duration,
            "campaign" : campaign,
            "pdays" : pdays,
            "previous" : previous,
            "job" : job,
            "marital" : marital,
            "education" : education,
            "default" : default,
            "housing" : housing,
            "loan" : loan,
            "contact" : contact,
            "month" : month,
            "day_of_week" : day_of_week,
            "poutcome" : poutcome,
            "empvarrate" : empvarrate,
            "conspriceidx" : conspriceidx,
            "consconfidx" : consconfidx,
            "euribor3m" : euribor3m,
            "nremployed" : nremployed
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] == "yes":
                st.warning("Predicted Subscription: YES")
            else:
                st.success("Predicted Subscription: NO")