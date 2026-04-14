import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 2: Load and Preprocess Data ---
print("Task 2: Loading and Preprocessing MNIST Dataset...")

# Step 1: Load the MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Step 2: Normalize the images (0-255 -> 0-1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Reshape to (Batch, Height, Width, Channels)
# Grayscale images have 1 channel; RGB would have 3.
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# --- Task 2: Define CNN Architecture ---
print("Defining CNN Architecture...")

model = models.Sequential([
    # First Convolutional Layer: Extracts 32 low-level features (edges)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), # Reduces image size by half
    
    # Second Convolutional Layer: Extracts 64 mid-level features
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Layer: Extracts high-level features
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Final Classification Layers
    layers.Flatten(), # Flatten 2D features into 1D vector
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes (digits 0-9)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Task 3: Train the CNN Model ---
print("\nTask 3: Starting Training (5 Epochs)...")
# We use validation_data to monitor performance on unseen data during training
model.fit(train_images, train_labels, epochs=5, 
          batch_size=64, validation_data=(test_images, test_labels))

# --- Task 4: Evaluate Results ---
print("\nTask 4: Evaluating Model Performance...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("-" * 30)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print("-" * 30)

if test_acc > 0.95:
    print("Goal Achieved: Accuracy is above the 95% target!")
