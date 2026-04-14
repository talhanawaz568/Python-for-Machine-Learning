import datetime

# --- Task 1: Parse Dates with the datetime Module ---
print("--- Task 1: Parsing and Current Time ---")

# 1.2 Parse a Date String
# strptime stands for "string parse time"
date_string = "2023-10-03"
parsed_date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
print(f"Parsed Date Object: {parsed_date}")

# 1.3 Obtain the Current Date and Time
current_datetime = datetime.datetime.now()
print(f"Current System Time: {current_datetime}")


# --- Task 2: Extract Components ---
print("\n--- Task 2: Extracting Components ---")

# 2.1 Extract Year, Month, Day
print(f"Year:  {parsed_date.year}")
print(f"Month: {parsed_date.month}")
print(f"Day:   {parsed_date.day}")

# 2.2 Extract Time Components (from the current time)
print(f"Hour:   {current_datetime.hour}")
print(f"Minute: {current_datetime.minute}")
print(f"Second: {current_datetime.second}")


# --- Task 3: Compute Time Differences ---
print("\n--- Task 3: Time Differences & Case Study ---")

# 3.1 Calculate Difference between two specific dates
another_date_string = "2023-10-10"
another_date = datetime.datetime.strptime(another_date_string, "%Y-%m-%d")

# Subtracting two datetime objects creates a 'timedelta' object
date_difference = another_date - parsed_date
print(f"Difference between {another_date_string} and {date_string}: {date_difference.days} days")

# 3.2 Case Study: Event Countdown
# Let's calculate days until the end of the year 2026
target_date_string = "2026-12-31"
target_date = datetime.datetime.strptime(target_date_string, "%Y-%m-%d")

countdown = target_date - current_datetime
print(f"Countdown to {target_date_string}: {countdown.days} days remaining.")

# Practical Bonus: Formatting for display
# strftime stands for "string format time"
print(f"Formatted Current Date: {current_datetime.strftime('%A, %B %d, %Y')}")
