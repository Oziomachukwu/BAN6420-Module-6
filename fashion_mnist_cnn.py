import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def build_cnn_model():
    # Create a sequential CNN model with 6 layers
    model = keras.Sequential([
        # First convolutional layer with 32 filters
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        # First max pooling layer
        keras.layers.MaxPooling2D((2,2)),
        # Second convolutional layer with 64 filters
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        # Second max pooling layer
        keras.layers.MaxPooling2D((2,2)),
        # Flatten layer to prepare for dense layers
        keras.layers.Flatten(),
        # Dense layer with 64 units
        keras.layers.Dense(64, activation='relu'),
        # Output layer with 10 units (one per class)
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with appropriate settings
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load and preprocess Fashion MNIST data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize pixel values and add channel dimension
    x_train = x_train[..., None].astype('float32') / 255  # [None] adds channel dimension
    x_test = x_test[..., None].astype('float32') / 255

    # Build and train the CNN model
    model = build_cnn_model()
    print("Training model...")
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Create full paths for saving assets
    model_path = os.path.join(SCRIPT_DIR, 'fashion_mnist_cnn.keras')
    test_images_path = os.path.join(SCRIPT_DIR, 'test_images.npy')
    test_labels_path = os.path.join(SCRIPT_DIR, 'test_labels.npy')

    # Save model and test data
    model.save(model_path)
    np.save(test_images_path, x_test)
    np.save(test_labels_path, y_test)
    print(f"Model and test data saved in: {SCRIPT_DIR}")

if __name__ == "__main__":
    main()