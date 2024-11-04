# import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential


class SoftmaxClassifier:
    """
    A classifier based on the softmax function for multi-class classification.

    Attributes:
        model (Sequential): A TensorFlow/Keras model comprising a hidden layer and a softmax output layer.

    Parameters:
        Nval (int): Number of features in the input dataset.
        Nclass (int): Number of classes to classify the input data into.
        Nhidden (int, optional): Number of neurons in the hidden layer. Defaults to 10.
    """

    def __init__(self, Nval, Nclass, Nhidden=10, Nlevel=1):
        """
        Initializes the SoftmaxClassifier with a specified number of input features, classes, and hidden neurons.

        The model consists of a dense hidden layer with ReLU activation and a dense output layer with softmax activation.

        Args:
            Nval (int): Number of features in the input dataset.
            Nclass (int): Number of classes for the output classification.
            Nhidden (int): Number of neurons in the hidden layer.
        """
        # Create the model
        TheModel = [Dense(units=Nhidden, activation="relu", input_shape=(Nval,))]

        for k in range(1, Nlevel):
            TheModel = TheModel + [
                Dense(units=Nhidden, activation="relu", input_shape=(Nhidden,))
            ]

        TheModel = TheModel + [
            Dense(
                units=Nclass
            ),  # The output layer with Nclass neurons (for Nclass classes)
            Softmax(),  # Softmax activation for classification
        ]
        self.model = Sequential(TheModel)

        # Model compilation
        self.model.compile(
            optimizer="adam",  # Adam optimizer
            loss="sparse_categorical_crossentropy",  # Loss function for Nclass-class classification
            metrics=["accuracy"],  # Evaluation metric: accuracy
        )

    def fit(self, x_train, y_train, epochs=10):
        """
        Trains the model on the provided dataset.

        Args:
            x_train (np.ndarray): Training data features, shape (num_samples, Nval).
            y_train (np.ndarray): Training data labels, shape (num_samples, ).
            epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        """
        # Train the model
        self.model.fit(x_train, y_train, epochs=epochs)

    def predict(self, x_train):
        """
        Predicts the class labels for the given input data.

        Args:
            x_train (np.ndarray): Input data for which to predict class labels, shape (num_samples, Nval).

        Returns:
            np.ndarray: Predicted class labels for the input data.
        """
        # Compute the prediction
        return self.model.predict(x_train)
