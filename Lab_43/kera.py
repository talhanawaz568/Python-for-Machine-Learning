import numpy as np
import os

# Suppress TensorFlow logging for a cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Task 1: Build a Simple Sequential Model ---
print("Task 1: Building the Neural Network...")

# Step 2: Create a Sequential Model
model = Sequential()

# Hidden Layer 1: 12 neurons, expecting 8 input features
# 'relu' is the industry standard for hidden layers to prevent vanishing gradients
model.add(Dense(12, input_dim=8, activation='relu'))

# Hidden Layer 2: 8 neurons
model.add(Dense(8, activation='relu'))

# Output Layer: 1 neuron with 'sigmoid' to output a probability between 0 and 1
model.add(Dense(1, activation='sigmoid'))

# Step 3: Compile the Model
# 'adam' adjusts the learning rate automatically
# 'binary_crossentropy' is used because we are predicting 0 or 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("✓ Model compiled successfully.")

# --- Task 2: Train the Model ---
print("\nTask 2: Preparing Data and Training...")

# Step 4: Generate Synthetic Data (100 samples, 8 features each)
np.random.seed(42) # For consistent results
X = np.random.rand(100, 8)
Y = np.random.randint(2, size=(100, 1))

# Step 5: Fit the Model
# epochs=150: The model sees the entire dataset 150 times
# batch_size=10: The model updates weights after every 10 samples
print("Training in progress... (150 epochs)")
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
print("✓ Training complete.")

# --- Task 3: Evaluate Performance ---
print("\nTask 3: Performance Evaluation")

# Step 6: Evaluate on Training Set
loss, accuracy = model.evaluate(X, Y, verbose=0)
print(f'Training Loss:     {loss:.4f}')
print(f'Training Accuracy: {accuracy:.4f}')

# Step 7: Evaluate on Test Performance (Unseen Data)
X_test = np.random.rand(20, 8)
Y_test = np.random.randint(2, size=(20, 1))
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

print("-" * 30)
print(f'Test Loss:         {test_loss:.4f}')
print(f'Test Accuracy:     {test_accuracy:.4f}')
print("-" * 30)

print("\n✓ Lab 43 Complete: The neural network is trained and evaluated.")
