import pandas as pd

df = pd.read_csv('imit_data.csv')
df = df[df['timestep'] == 0]

# print number of world idx
no_world_idx = len(df['world_idx'].unique())
print(no_world_idx)

# print number of success for each world idx
grouped = df.groupby(['world_idx'])['success'].any().astype(int)
success = grouped.sum()

print(grouped)
print("transformer success rate: {:.2f}%".format(float(success) / no_world_idx * 100))

print(df['success'].value_counts())