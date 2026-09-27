import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit.components.v1 as components


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Ethiopian Birr Recognition",
    page_icon="💵",
    layout="centered"
)


# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "5 Birr",
    "10 Birr",
    "50 Birr",
    "100 Birr",
    "200 Birr"
]


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/birr_mobilenetv2.keras"
    )

    return model


model = load_model()


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Convert pixels from 0-255 to 0-1
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get class with highest probability
    predicted_index = np.argmax(
        predictions[0]
    )

    # Get predicted denomination
    predicted_class = CLASS_NAMES[predicted_index]

    # Get confidence
    confidence = predictions[0][predicted_index]

    return (
        predicted_class,
        confidence,
        predictions[0]
    )


# ==========================================================
# AUTOMATIC VOICE OUTPUT
# ==========================================================

def speak_result(predicted_class):

    speech_text = (
        f"The predicted denomination is {predicted_class}"
    )

    components.html(
        f"""
        <script>

            // Create speech object
            const speech =
                new SpeechSynthesisUtterance(
                    "{speech_text}"
                );

            // Voice settings
            speech.lang = "en-US";
            speech.rate = 0.85;
            speech.pitch = 1.0;
            speech.volume = 1.0;

            // Stop previous speech
            window.speechSynthesis.cancel();

            // Speak prediction
            window.speechSynthesis.speak(speech);

        </script>
        """,
        height=0
    )


# ==========================================================
# APPLICATION TITLE
# ==========================================================

st.title(
    "🇪🇹 Ethiopian Birr Banknote Recognition"
)

st.write(
    "Use your camera or upload an image of an "
    "Ethiopian banknote. The AI model will identify "
    "the denomination."
)

st.divider()


# ==========================================================
# SELECT INPUT METHOD
# ==========================================================

st.subheader("📷 Choose Input Method")

input_method = st.radio(
    "How would you like to provide the banknote?",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)


# ==========================================================
# IMAGE VARIABLE
# ==========================================================

image = None


# ==========================================================
# UPLOAD IMAGE
# ==========================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a banknote image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Banknote",
            width=400
        )


# ==========================================================
# CAMERA INPUT
# ==========================================================

elif input_method == "Use Camera":

    st.write(
        "Place the banknote clearly in front of "
        "the camera and take a picture."
    )

    camera_image = st.camera_input(
        "Take a picture of the banknote"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        )

        st.image(
            image,
            caption="Captured Banknote",
            width=400
        )


# ==========================================================
# RECOGNITION
# ==========================================================

if image is not None:

    st.divider()

    recognize_button = st.button(
        "🔍 Recognize Banknote",
        use_container_width=True
    )

    if recognize_button:

        # --------------------------------------------------
        # Make Prediction
        # --------------------------------------------------

        with st.spinner(
            "Analyzing the banknote..."
        ):

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )


        # --------------------------------------------------
        # Display Prediction
        # --------------------------------------------------

        st.subheader(
            "🎯 Prediction"
        )

        st.success(
            f"💵 {predicted_class}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
        )


        # --------------------------------------------------
        # AUTOMATIC VOICE OUTPUT
        # --------------------------------------------------

        speak_result(
            predicted_class
        )


        # --------------------------------------------------
        # CONFIDENCE MESSAGE
        # --------------------------------------------------

        if confidence >= 0.90:

            st.info(
                "The model is highly confident in this prediction."
            )

        elif confidence >= 0.70:

            st.warning(
                "The model has moderate confidence. "
                "Consider taking another clearer image."
            )

        else:

            st.warning(
                "The model has low confidence. "
                "Try another image with better lighting "
                "and a clearer view of the banknote."
            )


        # --------------------------------------------------
        # PROBABILITY BREAKDOWN
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Prediction Probabilities"
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            st.write(
                f"**{class_name}:** "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Ethiopian Birr Banknote Recognition "
    "using MobileNetV2"
)