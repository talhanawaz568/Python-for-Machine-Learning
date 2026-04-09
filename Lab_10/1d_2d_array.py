import numpy as np

# Creating a 1D array
one_dimensional_array = np.array([1, 2, 3, 4, 5])
print("1D array:", one_dimensional_array)

# Creating a 2D array
two_dimensional_array = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:")
print(two_dimensional_array)


# Array addition
array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])
result = array_a + array_b
print("Array addition:", result)
# Array multiplication
result = array_a * array_b
print("Array multiplication:", result)

# Accessing elements
print("First element:", one_dimensional_array[0])
print("Last element:", one_dimensional_array[-1])


# Slicing a 1D array
print("Slice 1D array:", one_dimensional_array[1:4])

# Slicing a 2D array
print("Slice 2D array:\n", two_dimensional_array[0:2, 1:3])


