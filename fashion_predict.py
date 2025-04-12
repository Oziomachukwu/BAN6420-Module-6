import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Class names for Fashion MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def load_assets():
    # Load model and test data from script directory
    model = keras.models.load_model(os.path.join(SCRIPT_DIR, 'fashion_mnist_cnn.keras'))
    x_test = np.load(os.path.join(SCRIPT_DIR, 'test_images.npy'))
    y_test = np.load(os.path.join(SCRIPT_DIR, 'test_labels.npy'))
    return model, x_test, y_test

def plot_predictions(model, x_test, y_test, num_images=3):
    # Randomly select images from test set
    random_indices = np.random.choice(len(x_test), num_images, replace=False)
    images = x_test[random_indices]
    true_labels = y_test[random_indices]
    
    # Generate predictions
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create plot with predictions
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')  # Remove channel dimension for plotting
        plt.title(f"Predicted: {CLASS_NAMES[predicted_labels[i]]}\nActual: {CLASS_NAMES[true_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load model and data
    model, x_test, y_test = load_assets()
    
    # Make and display predictions
    print("Generating predictions for 3 random images...")
    plot_predictions(model, x_test, y_test)

if __name__ == "__main__":
    main()