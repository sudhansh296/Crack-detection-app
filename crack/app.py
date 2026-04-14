import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Crack Detection AI", page_icon="🏗️")

# 2. Model Load Karein
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('my_cnn_model.h5')

model = load_my_model()

# 3. Headers
st.title("🏗️ Surface Crack Detection AI")
st.write("Upload a surface image or take a picture to check for structural cracks.")

# 4. UI Tabs (Upload vs Camera)
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Use Camera"])

image_data = None # Variable to store the final image

# Tab 1: Upload Option
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file

# Tab 2: Camera Option
with tab2:
    camera_file = st.camera_input("Take a picture of the surface")
    if camera_file is not None:
        image_data = camera_file

# 5. Image Processing & Prediction
if image_data is not None:
    # Image open aur RGB convert (4-channel transparent image error solve karne ke liye)
    image = Image.open(image_data)
    image = image.convert('RGB')
    
    st.image(image, caption='Image to Analyze', use_column_width=True)
    
    # Preprocessing (120x120)
    img = image.resize((120, 120)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Analyze Button
    if st.button("Detect Crack", type="primary"):
        with st.spinner('Analyzing surface...'):
            prediction = model.predict(img_array)
            
            # Confidence Calculation
            confidence = float(prediction[0][0])
            
            # Percentage mein convert kar rahe hain (e.g., 0.9654 * 100 = 96.54%)
            confidence_percentage = confidence * 100 
            
            # Result Logic with Progress Bar
            if confidence > 0.5:
                # Agar crack hai
                st.error(f"⚠️ Crack Detected! ({confidence_percentage:.1f}% Sure)")
                st.progress(confidence) # Visual Bar (Isme 0 se 1 ki value chahiye hoti hai)
            else:
                # Agar crack nahi hai
                safe_confidence = 1.0 - confidence
                safe_confidence_percentage = safe_confidence * 100
                st.success(f"✅ Surface is Smooth/No Crack ({safe_confidence_percentage:.1f}% Sure)")
                st.progress(safe_confidence) # Visual Bar