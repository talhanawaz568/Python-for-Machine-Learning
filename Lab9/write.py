output_file = "output.txt"
with open(output_file, 'w') as file:
    file.write("This is a new file.\n")
    file.write("We are writing data to it.\n")
with open(output_file, 'a') as file:
    file.write("Appending a new line of text.\n")
