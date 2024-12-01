import pandas as pd

df = pd.read_csv('new_imit_data.csv')

# group using cumsum timestep = 0
df['group'] = (df['timestep'] == 0).cumsum()

# count groups that are successful
successful_groups = df.groupby('group').apply(lambda x: (x['success'] == 1).any()).sum()

print("No. groups: ", len(df['group'].unique()))
print(f"No. success: {successful_groups}")
print(f"Success rate: {successful_groups / len(df['group'].unique())}")