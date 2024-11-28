import os
import streamlit as st
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Directory to save images
image_folder = 'uploaded_images/'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Load the model
model = load_model('best_model.keras')

# Function to delete all images in the folder
def clear_images():
    for file in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Top Header
st.markdown(
    """
    <style>
    .header-text {
        text-align: center; 
        color: white; 
        font-size: 40px; 
        margin-top: -50px; 
    }
    .sub-header-text {
        text-align: center; 
        color: white; 
        font-size: 24px; 
        margin-top: -20px;
    }
    </style>
    <div class="header-text">
        One Stop Solution for Diabetic Solution
    </div>
    <div class="sub-header-text">
        Have AI-Based Insights for your daily life using our tool
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Select Category")

# Dropdown for States or Union Territories
category = st.sidebar.selectbox("Choose an option:", ["State", "Union Territory"])

# Define States and Union Territories
states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", 
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", 
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", 
    "West Bengal"
]

union_territories = [
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi (NCT)", 
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

# Dropdown to select State or Union Territory based on the selected category
if category == "State":
    state = st.sidebar.selectbox("Select a State", states, key="state")
elif category == "Union Territory":
    union_territory = st.sidebar.selectbox("Select a Union Territory", union_territories, key="union_territory")

# Height and Weight inputs
height = st.sidebar.text_input("Height (in cm):", key="height")
weight = st.sidebar.text_input("Weight (in kg):", key="weight")

# Date of Birth input with calendar
dob = st.sidebar.date_input("Date of Birth", key="dob")

# Allergies and Medications input
allergies = st.sidebar.text_input("Allergies (if any):", key="allergies")
medications = st.sidebar.text_input("Current Medications (if any):", key="medications")

# Image Upload
image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image")

# Submit Button
submit_button = st.sidebar.button("Submit")

# Refresh Data Button
refresh_button = st.sidebar.button("Refresh Data")

# Function to calculate age from date of birth
def calculate_age(birth_date):
    today = datetime.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# Form validation for height and weight (positive decimal values)
def validate_input():
    try:
        height_float = float(height)  # Convert to float for decimal values
        weight_float = float(weight)
        if height_float <= 0 or weight_float <= 0:
            raise ValueError("Height and weight must be positive decimal numbers.")
        return True
    except ValueError:
        st.sidebar.error("Height and weight need to be numbers.")
        return False

# Function to predict class of uploaded image
def predict_image_class(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as per your model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image (if your model requires normalization)
    img_array = img_array / 255.0  # Comment out if not needed

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Assuming single image input
    return predicted_class

# Action on Submit
if submit_button:
    if validate_input():
        # Get values
        selected_state = state if category == "State" else union_territory
        age = calculate_age(dob)  # Calculate age from Date of Birth
        height_float = float(height)
        weight_float = float(weight)

        # Save the image if uploaded
        image_path = None
        if image is not None:
            image_path = os.path.join(image_folder, f"{selected_state}_{dob}.jpg")
            with open(image_path, "wb") as img_file:
                img_file.write(image.getbuffer())

            # Predict the class for the uploaded image
            predicted_class = predict_image_class(image_path)

            # Display prediction result
            st.markdown(f"<div style='text-align: center; font-size: 20px; color: white;'>Predicted Class: {predicted_class}</div>", unsafe_allow_html=True)

        # Display output
        st.markdown(f"""
        <div style="text-align: center; font-size: 20px; color: white;">
            <p>The person is from <b>{selected_state}</b> with height <b>{height_float} cm</b> 
            and weight <b>{weight_float} kg</b>, age <b>{age}</b>, on medication <b>{medications}</b> 
            with allergies <b>{allergies}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        # Display uploaded image if available
        if image_path:
            st.markdown("<div style='text-align: center; font-size: 20px; color: white;'>Below is an image of the eye:</div>", unsafe_allow_html=True)
            st.image(image_path, caption="Uploaded Eye Image", use_column_width=True)
    else:
        st.sidebar.warning("Please correct the errors before submitting.")

# Action on Refresh Data
if refresh_button:
    # Reset all variables
    st.session_state.clear()
    clear_images()  # Delete all images
    st.sidebar.success("Data has been refreshed.")
