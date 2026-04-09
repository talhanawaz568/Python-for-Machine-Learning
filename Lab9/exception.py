try:
    with open("non_existent_file.txt", 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("The file you are trying to read does not exist.")
