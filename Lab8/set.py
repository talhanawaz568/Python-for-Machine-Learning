fruits = {"apple", "banana", "cherry"}

fruits.add("orange")


# Removing an element
fruits.remove("banana") # Raises an error if not found
# Alternatively, use discard which doesn't raise an error
fruits.discard("banana")


tropical_fruits = {"pineapple", "mango", "papaya", "apple"}


all_fruits = fruits.union(tropical_fruits)
print("All fruits:", all_fruits)

common_fruits = fruits.intersection(tropical_fruits)
print("Common fruits:", common_fruits)
