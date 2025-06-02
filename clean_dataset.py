import pandas as pd

# Load the original dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny\\emotions.csv")

# Check original class distribution
print("Original Distribution:\n", df['label'].value_counts())

# Downsample the majority class(es)
min_count = df['label'].value_counts().min()
balanced_df = df.groupby('label').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Check new class distribution
print("\nBalanced Distribution:\n", balanced_df['label'].value_counts())

# Save the balanced dataset
balanced_df.to_csv("balanced_emotion_dataset.csv", index=False)
print("\nBalanced dataset saved as 'balanced_emotion_dataset_2.csv'")
