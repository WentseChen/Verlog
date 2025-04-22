import pandas as pd

# Define the path to the Parquet file
file_path = '/u/wchen11/data/gsm8k/train.parquet'  # Replace 'username' with your actual username

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# print data in details
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns) 
# DataFrame columns: Index(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'], dtype='object')
print("First few rows of the DataFrame:")
print(df.head())
# print data for each column
print("Data source:", df['data_source'])
print("Prompt:", df['prompt'])
print("Ability:", df['ability'])
print("Reward model:", df['reward_model'])
print("Extra info:", df['extra_info'])