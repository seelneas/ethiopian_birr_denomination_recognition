import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Ethiopian Birr Recognition",
    page_icon="💵",
    layout="centered"
)


# --------------------------------------------------
# Class Names
# --------------------------------------------------

CLASS_NAMES = [
    "5 Birr",
    "10 Birr",
    "50 Birr",
    "100 Birr",
    "200 Birr"
]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/birr_mobilenetv2.keras"
    )


model = load_model()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_image(image):
    
    image = image.convert("RGB")
    
    image = image.resize((224, 224))
    
    image_array = np.array(image)
    
    image_array = image_array.astype("float32") / 255.0
    
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array, verbose=0)
    
    predicted_class = np.argmax(predictions[0])
    
    confidence = predictions[0][predicted_class]
    
    return (
        CLASS_NAMES[predicted_class],
        confidence,
        predictions[0]
    )


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

st.title("🇪🇹 Ethiopian Birr Banknote Recognition")

st.write(
    "Upload an image of an Ethiopian banknote and "
    "the AI model will predict its denomination."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a banknote image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Banknote",
        width=400
    )

    st.divider()

    if st.button("🔍 Recognize Banknote"):

        predicted_class, confidence, probabilities = predict_image(image)

        st.subheader("Prediction")

        st.success(
            f"💵 {predicted_class}"
        )

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.divider()

        st.subheader("Prediction Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):
            st.write(
                f"{class_name}: {probability * 100:.2f}%"
            )