import pandas as pd

# Load each dataset
dataset1 = pd.read_csv('feed.csv')  # Replace with your actual file name
dataset2 = pd.read_csv('phishing.csv')  # Replace with your actual file name
dataset3 = pd.read_csv('PhishingData.csv')  # Replace with your actual file name
dataset4 = pd.read_csv('Training Dataset.csv')  # Replace with your actual file name

# Combine the datasets into one DataFrame
combined_dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

# Display the first few rows of the combined dataset
print(combined_dataset.head())
