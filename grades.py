import json

# Initialize a set to store unique grades
unique_grades = set()

# List of file paths
file_paths = [
    '/users/plessas/Downloads/BA.txt', 
    '/users/plessas/Downloads/MSFT.txt', 
    '/users/plessas/Downloads/AAPL.txt', 
    '/users/plessas/Downloads/NVDA.txt', 
    '/users/plessas/Downloads/UBER.txt',
    '/users/plessas/Downloads/INTC.txt',
    '/users/plessas/Downloads/TSLA.txt'
]

# Process each file
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        try:
            data = json.loads(content)
            for entry in data:
                unique_grades.add(entry['previousGrade'].lower())
                unique_grades.add(entry['newGrade'].lower())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            continue

# Sort and print the unique grades
sorted_unique_grades = sorted(unique_grades)

print(sorted_unique_grades)
print(len(sorted_unique_grades))
