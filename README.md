# Vegetable Image Classification Project

This is a Vegetable Image Classification Project that utilizes the TensorFlow framework and Streamlit for creating a user-friendly web application to classify various vegetables based on their images. For the present time this project can detect only 15 items of vegitable. This README file provides an overview of the project, its components, and instructions on how to set it up and run it.

## Project Overview

The Vegetable Image Classification Project aims to classify different types of vegetables from images. It leverages the power of deep learning using TensorFlow to build and train a convolutional neural network (CNN) model. This model is then deployed as a user-friendly web application using Streamlit, allowing users to upload images of vegetables for classification.

### Project Components

The project consists of the following major components:

1. **Data Collection and Preparation**: The dataset of vegetable images is collected and preprocessed. This involves data augmentation, splitting into training and testing sets, and converting images into a format suitable for training.
Data link : (https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

2. **Model Development**: A deep learning model, typically a Convolutional Neural Network (CNN), is designed, trained, and fine-tuned using the prepared dataset. TensorFlow is used for model development and training.

3. **Streamlit Web Application**: The trained model is integrated into a Streamlit web application. Streamlit is used to create a simple and interactive interface for users to upload images and get real-time predictions.

4. **Deployment**: The Streamlit app is deployed to a server or cloud platform so that it can be accessed by users over the internet.

5. **Documentation and Testing**: Comprehensive documentation and testing are performed to ensure that the project functions as expected.

## Project Setup

Follow these steps to set up and run the Vegetable Image Classification Project on your local machine:

### Prerequisites

- Python 3.x installed on your system.
- TensorFlow and other required Python libraries. You can install them using `pip`:

```bash
pip install tensorflow streamlit
```

### Clone the Repository

Clone the project repository to your local machine using Git:

```bash
git clone https://github.com/me-sajal/Vegetable-Image-Classification.git
cd Vegetable-Image-Classification
```

### Training the Model

1. Place your vegetable dataset in a directory, e.g., `data/vegetables`.

2. Run the script to train the model:

```bash
python train_model.py --dataset data/vegetables --output model/vegetable_classifier
```

This will train the model using the provided dataset and save it in the `model/vegetable_classifier` directory.

### Running the Streamlit App

To run the Streamlit web application, use the following command:

```bash
streamlit run main.py
```

This will start the Streamlit app locally, and you can access it by opening a web browser and navigating to `http://localhost:8501`.

## Usage

1. Open the Streamlit web application in your web browser by following the instructions above.

2. Use the app's interface to upload an image of a vegetable.

3. Click the "Classify" button to see the predicted vegetable category.

4. Explore the app's features, and enjoy classifying vegetable images!

## Project Structure

The project structure is organized as follows:

- `data/`: (https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- `model/`: Directory where the trained model is saved.
- `main.py`: The Streamlit web application script.
- `vegetable_image_classification.ipynb`: Script for training the vegetable classification model.
- `Evulaion Image`: Contains the evulated images like confusion matrix, accuracy and loss.
- `Image to test`: Contains some images of vegitables that are said to be tested.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Acknowledgments

- Thanks to the TensorFlow and Streamlit communities for providing excellent tools and resources for deep learning and web application development.

Feel free to contribute to and customize this project as needed. Happy classifying vegetables!
