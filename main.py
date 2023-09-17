import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the model
model = tf.keras.models.load_model('model/model.h5')

# Class labels
labels = {0: 'Bean',
          1: 'Bitter Gourd',
          2: 'Bottle Gourd',
          3: 'Brinjal',
          4: 'Broccoli',
          5: 'Cabbage',
          6: 'Capsicum',
          7: 'Carrot',
          8: 'Cauliflower',
          9: 'Cucumber',
          10: 'Papaya',
          11: 'Potato',
          12: 'Pumpkin',
          13: 'Radish',
          14: 'Tomato'}


def predict(image):
    img = tf.keras.utils.load_img(image, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display output size of 120 x 120
    st.image(img, caption='Uploaded Image.', width=400)

    # Calculate prediction accuracy
    prediction_accuracy = np.max(prediction)

    # Display prediction accuracy
    st.write(f"Prediction accuracy: {prediction_accuracy:.2f}")

    return predicted_class


def  main():
    st.title("Vegetable Classifier")

    uploaded_image = st.file_uploader(
        "Upload an image  [Bean, Bitter Gourd, Bottle Gourd, Brinjal,Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato]",
        type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Make prediction
        predicted_class = predict(uploaded_image)
        st.write(f"Prediction: {labels[predicted_class]}")


if __name__ == '__main__':
    main()