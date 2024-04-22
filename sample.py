import os


# def list_files_folders(directory, exclude_folders):
#     files_folders = []
#     for root, dirs, files in os.walk(directory):
#         # Exclude specified folders
#         dirs[:] = [d for d in dirs if d not in exclude_folders]
#         for file in files:
#             files_folders.append(os.path.join(root, file))
#         for folder in dirs:
#             files_folders.append(os.path.join(root, folder))
#     return files_folders


# project_directory = "."  # Replace this with your project directory path
# exclude_folders = ["env", "_pycache_"]  # Add folders to exclude here
# files_folders = list_files_folders(project_directory, exclude_folders)
# for item in files_folders:
#     print(item)


import os
import pandas as pd

# List of folders containing text files
folders = ["diets"]  # Update with your folder names

# Initialize an empty list to store data
data = []

# Loop through each folder
for folder in folders:
    # Get list of files in the folder
    files = os.listdir(folder)

    # Loop through each file in the folder
    for file in files:
        # Read content of each file
        with open(os.path.join(folder, file), "r") as f:
            content = f.read()
            data.append({"filename": file, "content": content})
        f.close()

# Create DataFrame from the collected data
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv("combined_data.csv", index=False)
