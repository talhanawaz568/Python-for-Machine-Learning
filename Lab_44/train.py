import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Capture Training History ---
print("Task 1: Loading Data and Building Model...")

# 1.4. Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# 1.3. Create and Compile the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') # 10 classes for clothing types
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and capture history
print("\nStarting Training (10 Epochs)...")
# validation_split=0.2 sets aside 20% of data to check for overfitting during training
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2, verbose=1)

# --- Task 2: Plot Loss and Accuracy ---
print("\nTask 2: Generating Plots...")

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# 2.2. Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) # Create a side-by-side plot
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 2.3. Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the plots to a file since we are working in a terminal environment
plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Success: Training history plots saved as 'training_history.png'")

# --- Task 3: Analysis Summary ---
print("\n--- Task 3: Convergence Analysis ---")
final_train_acc = acc[-1]
final_val_acc = val_acc[-1]

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

if (final_train_acc - final_val_acc) > 0.05:
    print("ANALYSIS: Potential Overfitting detected (Training accuracy is significantly higher than Validation).")
else:
    print("ANALYSIS: Model appears to be generalizing well.")
