import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("Model_2.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.array([image])
    image = image / 255.0
    predictions = model.predict(image)
    return predictions  # return the full predictions array

# Sidebar
st.sidebar.title("🌈 Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "📖 About Project", "🔍 Prediction"])

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5em;
        color: #FFD700;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .sub-header {
        font-size: 1.8em;
        color: #FF6347;
        font-family: 'Arial', sans-serif;
        margin-top: 1em;
    }
    .dataset-section {
        background-color: #2E2E2E;
        color: #F0E68C;
        padding: 15px;
        border-radius: 8px;
    }
    .content-box {
        font-size: 1.2em;
        background-color: #2F4F4F;
        color: #F8F9FA;
        border-left: 5px solid #FF4500;
        padding: 10px;
        margin-bottom: 1em;
    }
    .labels-box {
        background-color: #3CB371;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Page
if app_mode == "🏠 Home":
    st.markdown('<div class="main-header">🍎🍊 FRUITS & VEGETABLES RECOGNITION SYSTEM 🥦🍅</div>', unsafe_allow_html=True)
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "📖 About Project":
    st.markdown('<div class="main-header">📖 About Project</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">📊 About Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dataset-section">
            <p>This dataset contains images of the following food items:</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="labels-box">', unsafe_allow_html=True)
    st.code("""
    Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
    Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">📁 Content</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content-box">
        <p>This dataset contains three folders:</p>
        <ul>
        <li>1. <b>Train:</b> 100 images each</li>
        <li>2. <b>Test:</b> 10 images each</li>
        <li>3. <b>Validation:</b> 10 images each</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "🔍 Prediction":
    st.markdown('<div class="main-header">🔍 Model Prediction</div>', unsafe_allow_html=True)

    test_image = st.file_uploader("📂 Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Predict button
        if st.button("🔮 Predict"):
            st.snow()
            st.write("Our Prediction is processing...")

            predictions = model_prediction(test_image)
            result_index = np.argmax(predictions)
            probability = np.max(predictions)

            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]  # remove any newlines

            # Show prediction and probability
            st.success(f"✨ The model predicts it's a {label[result_index]} with a probability of {probability:.2f}.")

            # Display the uploaded image
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
            st.image(image, caption=f"Predicted: {label[result_index]} with prob: {probability:.2f}", use_column_width=True)
    