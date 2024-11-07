# number_detection
 My first machine learning project uses the MNIST dataset to develop a model for digit recognition. Through data preprocessing, model training, and evaluation, this project focuses on accurately identifying handwritten numbers from 0 to 9. It serves as a foundation for understanding neural networks and basic image classification techniques.

1)Imports and Data Loading:

tensorflow is imported, and the MNIST dataset is loaded.
(x_train, y_train), (x_test, y_test) = mnist.load_data() loads the training and testing data, which contain grayscale images of handwritten digits (0-9).

2)Data Visualization:

matplotlib.pyplot is used to display sample images from the training set (x_train[0]) to understand the dataset visually.
Data Preprocessing:

Data normalization scales pixel values to the range [0, 1] by dividing by 255. This step ensures that inputs are normalized, which can improve model convergence.
Reshaping (np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)) adds a channel dimension to the data (28x28 to 28x28x1), which is required by the convolutional layers.

3)Model Architecture (Convolutional Neural Network):

A Sequential model is constructed with the following layers:
Conv2D Layers: Three convolutional layers are added with 64 filters each, using a 3x3 kernel. ReLU activation is applied after each convolution.
MaxPooling2D Layers: Pooling layers follow each Conv2D layer to reduce dimensionality and computation, using a 2x2 pool size.
Flatten Layer: Converts the 3D output to 1D for fully connected layers.
Dense Layers: Two dense layers with 64 and 32 units, followed by ReLU activation.
Output Layer: The final dense layer has 10 units (one for each digit class) with softmax activation to output class probabilities.
model.summary() provides an overview of the model architecture.

4)Model Compilation and Training:

The model is compiled with a loss function (sparse_categorical_crossentropy), optimizer (adam), and accuracy as the metric.
The model is trained on 60,000 samples, with a validation split of 30%, over 5 epochs.

5)Model Evaluation:

The model’s performance on the test set is evaluated using model.evaluate().

6)Prediction:

Predictions are generated for test images using model.predict().
np.argmax(predictions[0]) is used to determine the predicted class with the highest probability.

7)Custom Image Prediction:

The notebook includes steps to load a custom image (paint5.png), convert it to grayscale, normalize it, reshape it, and pass it through the model for prediction. This demonstrates the model's usage on new, external data.


8)This project structure covers:

Data Preprocessing: Normalization and reshaping.
Model Building: A CNN architecture suited for image classification.
Training: Compiling and fitting the model.
Evaluation: Testing accuracy and loss.
Real-world Application: Extending the model to predict new handwritten digits from custom images.