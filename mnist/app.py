import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.title("🔢 MNIST Digit Classifier")

# Load model
model = joblib.load("model.pkl")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # MNIST uses black background + white digit
    # Invert if image has a white background
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize pixels
    image_array = image_array.astype("float32") / 255.0

    # Keras MNIST model expects:
    # (batch_size, 28, 28)
    image_array = image_array.reshape(1, 28, 28)

    # Prediction
    predictions = model.predict(image_array)

    prediction = np.argmax(predictions, axis=1)[0]

    st.success(f"Predicted Digit: **{prediction}**")

    # Show probabilities
    st.subheader("Prediction probabilities")

    probabilities = predictions[0]

    for digit, probability in enumerate(probabilities):
        st.write(f"{digit}: {probability:.2%}")
