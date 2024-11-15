# Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:
Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:
Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
X_train = X_train.reshape(-1, 784)  # Flatten the images
X_test = X_test.reshape(-1, 784)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Input layer with 128 neurons
    Dense(64, activation='relu'),                       # Hidden layer
    Dense(10, activation='softmax')                     # Output layer for 10 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test).argmax(axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()

misclassified_idx = np.where(y_pred != y_test)[0]
for i in range(5):  # Show first 5 misclassified examples
    idx = misclassified_idx[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {y_test[idx]}, Predicted: {y_pred[idx]}")
    plt.show()
```

## Output:
### Test Accuracy:

![1](https://github.com/user-attachments/assets/916c94ed-90f4-48d4-8fea-69510e744998)

### Confusion Matrix:

![2](https://github.com/user-attachments/assets/e2daecc8-d244-48d5-9170-bd07ba8b769d)

### Training and Validation Accuracy Plot:

![3](https://github.com/user-attachments/assets/010ed33c-4d2f-4e01-a297-95afb3dd40b2)

###  Misclassified Examples:

![4](https://github.com/user-attachments/assets/0176f243-1331-4849-8a57-78235970a07b)
![5](https://github.com/user-attachments/assets/5949ee23-cccc-447d-93c3-f14381e599c8)
![6](https://github.com/user-attachments/assets/7eab8aba-2f54-4995-9ee8-639e9826d781)
![7](https://github.com/user-attachments/assets/a4b6b852-6281-4038-94d5-809f4c3f4baf)
![8](https://github.com/user-attachments/assets/d5e07b4e-5181-4a51-bd96-cd3e8686bb97)


## Result:

Hence, to Build a Multilayer Perceptron (MLP) to classify handwritten digits in python is done successfully.


