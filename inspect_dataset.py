import pandas as pd
df = pd.read_csv(r"C:\Users\Momo\PycharmProjects\hate_speech_bot\data\hate_speech_data.csv")  # Corrected path
print("Dataset Summary:")
print(df["sentiment"].describe())
print("\nLabel Distribution (Hate Speech: <= -1.0, Normal: > -1.0 and < 1.0, Offensive: >= 1.0):")
print(df["sentiment"].value_counts(bins=[-float("inf"), -1.0, 1.0, float("inf")]))