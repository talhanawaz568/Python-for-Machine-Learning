file_path = "sample.txt"
with open(file_path, 'r') as file:
    for line in file:
        print(line.strip())
