inventory = {'apples': 15, 'bananas': 5, 'oranges': 12}

for fruit, quantity in inventory.items():
    if quantity < 10:
        print(f"Restock {fruit}: Quantity is {quantity}")
    else:
        print(f"{fruit} are sufficiently stocked")
