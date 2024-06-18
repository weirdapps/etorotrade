import os
import re

def extract_imports_from_file(file_path):
    imports = set()
    with open(file_path, 'r') as file:
        content = file.read()
        import_statements = re.findall(r'^\s*(import|from)\s+(\S+)', content, re.MULTILINE)
        for statement in import_statements:
            imports.add(statement[1].split('.')[0])
    return imports

def extract_imports_from_folder(folder_path, exclude_folders=None):
    if exclude_folders is None:
        exclude_folders = []
    imports = set()
    for root, _, files in os.walk(folder_path):
        if any(excluded in root for excluded in exclude_folders):
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports.update(extract_imports_from_file(file_path))
    return imports

def extract_requirements(requirements_file_path):
    with open(requirements_file_path, 'r') as file:
        requirements = {line.strip().split('==')[0] for line in file if line.strip() and not line.startswith('#')}
    return requirements

def compare_requirements(folder_path, requirements_file_path, exclude_folders=None):
    imported_modules = extract_imports_from_folder(folder_path, exclude_folders)
    required_modules = extract_requirements(requirements_file_path)

    not_in_requirements = imported_modules - required_modules
    not_in_imports = required_modules - imported_modules

    return not_in_requirements, not_in_imports

# Specify the paths
folder_path = '/Users/plessas/OneDrive/Documents/SourceCode/python/trade/'
requirements_file_path = 'requirements.txt'
exclude_folders = ['myenv']  # Add any other folders you want to exclude

not_in_requirements, not_in_imports = compare_requirements(folder_path, requirements_file_path, exclude_folders)

print("Modules imported in .py files but not in requirements.txt:")
print(not_in_requirements)

print("\nModules in requirements.txt but not imported in any .py files:")
print(not_in_imports)