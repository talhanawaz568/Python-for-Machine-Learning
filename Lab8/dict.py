# Create a dictionary for a book
book = {
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "year_published": 1960
}

author_name = book["author"]
print(f"The author of the book is: {author_name}")


book["year_published"] = 1961
book["genre"] = "Fiction"


print(book)
