import numpy as np

# Creating a 3x4 NumPy array
array_2d = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])
print("Original Array:\n", array_2d)

# Extract elements from 2nd and 3rd column
sliced_array = array_2d[:, 1:3]
print("Sliced Array (columns 2 to 3):\n", sliced_array)

# Extract every second element from the array
advanced_slicing = array_2d[::2, ::2]
print("Advanced Slicing:\n", advanced_slicing)

# Add 10 to all elements in the array
added_array = array_2d + 10
print("Array after addition:\n", added_array)

# Multiply each element by 2
multiplied_array = array_2d * 2
print("Array after multiplication:\n", multiplied_array)

# Adding a 1D array to each row of a 2D array
added_with_broadcasting = array_2d + np.array([1, 0, 1, 0])
print("Array after broadcasting:\n", added_with_broadcasting)

reshaped_array = np.arange(12).reshape(3, 4)
print("Reshaped Array:\n", reshaped_array)


flattened_array = array_2d.flatten()
print("Flattened Array:\n", flattened_array)

transposed_array = array_2d.T
print("Transposed Array:\n", transposed_array)
