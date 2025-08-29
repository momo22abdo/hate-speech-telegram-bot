from datasets import load_dataset
import pandas as pd
import os

# Define project directory
project_dir = r"C:\Users\Momo\PycharmProjects\hate_speech_bot"
output_file = os.path.join(project_dir, "hate_speech_data.csv")

# Load the dataset from Hugging Face
print("Downloading Measuring Hate Speech dataset...")
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")

# Convert to pandas DataFrame
df = dataset.to_pandas()

# Select relevant columns: 'text' and 'hate_speech_score' (renamed to 'sentiment')
df = df[["text", "hate_speech_score"]].rename(columns={"hate_speech_score": "sentiment"})

# Save to CSV
df.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")