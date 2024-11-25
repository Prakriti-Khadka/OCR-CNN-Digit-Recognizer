import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image  # To load the handwritten image
import numpy as np  # To handle image array operations
import pytesseract  # To perform OCR on the image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: A tuple containing the training and testing datasets.
               - Training data: (train_images, train_labels)
               - Testing data: (test_images, test_labels)
               Images are reshaped to (28, 28, 1) and normalized to a range of [0, 1].
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    """
    Build a Convolutional Neural Network (CNN) model for MNIST digit classification.

    Returns:
        tf.keras.Model: The constructed CNN model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train_model(model, train_images, train_labels, test_images, test_labels):
    """
    Train the given model on the training data and validate it on the test data.

    Args:
        model (tf.keras.Model): The CNN model to train.
        train_images (numpy.ndarray): Training images.
        train_labels (numpy.ndarray): Labels corresponding to training images.
        test_images (numpy.ndarray): Testing images.
        test_labels (numpy.ndarray): Labels corresponding to testing images.
    """
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=0, batch_size=64,
              validation_data=(test_images, test_labels), verbose=0)


def evaluate_model(model, test_images, test_labels):
    """
    Evaluate the trained model's performance on the test data.

    Args:
        model (tf.keras.Model): The trained CNN model.
        test_images (numpy.ndarray): Testing images.
        test_labels (numpy.ndarray): Labels corresponding to testing images.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return test_acc 

def extract_text_from_image(image_path):
    """
    Extract text from an image using Optical Character Recognition (OCR).

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The text extracted from the image.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def save_text_to_file(text, classification, file_path):
    """
    Save extracted text and its classification to a file.

    Args:
        text (str): The text extracted from the image.
        classification (str): The classification of the text (Handwritten or Typed).
        file_path (str): The path of the output file to save the text and classification.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n\n\n\n')
        f.write(f"The extracted text is classified as: {classification}\n")


def classify_text_type(text):
    """
    Determine whether the extracted text is handwritten or typed.

    Args:
        text (str): The text extracted from the image.

    Returns:
        str: The classification result ('Handwritten' or 'Typed').
    """

    if any(char.isdigit() for char in text) or len(text) > 50:  
        return "Typed"
    else:
        return "Handwritten"

if __name__ == "__main__":

    # Load and preprocess the MNIST data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # Build and train the model
    model = build_model()
    train_model(model, train_images, train_labels, test_images, test_labels)

    # Evaluate the model and get the accuracy
    test_accuracy = evaluate_model(model, test_images, test_labels)

    # Extract text from an image which is in the specified file
    handwritten_image_path = 'image2.png' 
    extracted_text = extract_text_from_image(handwritten_image_path)

    # Save the extracted text and classification to the specified file 
    output_file_path = 'imageText.txt' 
    text_classification = classify_text_type(extracted_text)
    save_text_to_file(extracted_text, text_classification, output_file_path)

    # Print the essential output in the output file with accuracy level
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"The extracted text has been saved to: {output_file_path}")


