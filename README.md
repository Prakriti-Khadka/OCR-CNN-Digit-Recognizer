# OCR-CNN-Digit-Recognizer
A Python-based project that combines Optical Character Recognition (OCR) and Convolutional Neural Networks (CNN) to extract and classify text from images. This project recognizes handwritten or typed text and performs digit classification using the MNIST dataset. Ideal for tasks requiring text extraction, handwriting detection.

Limitations : 

This project leverages Optical Character Recognition (OCR) and a Convolutional Neural Network (CNN) trained on the MNIST dataset. While the model achieves a high accuracy on the MNIST test data, it is not 100% accurate and may produce errors in certain scenarios, including:
Poor image quality or noisy backgrounds.
Unusual fonts or handwriting styles.
Model Constraints: The CNN model is trained exclusively on the MNIST dataset, which consists of grayscale images of handwritten digits (0-9). It may not generalize well to non-MNIST data or digits with significant variations.
Handwriting vs. Typed Classification: The simple heuristic used to distinguish between handwritten and typed text may not always provide accurate results, particularly with mixed or stylized content in the image.

Notes : 

For OCR accuracy, ensure the input images are clear and properly preprocessed.
For digit classification, the model is limited to digits within the MNIST dataset's domain and may not perform well with custom data or other character sets.
Always validate the outputs for critical applications.
