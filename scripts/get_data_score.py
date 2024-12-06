import pandas as pd

df = pd.read_csv('inspection_data/cnn_imit_data_50Hz.csv')

df = df[df['timestep'] == 0]

# take the first data of every world_idx
df = df.groupby('world_idx').first().reset_index()

# only take multiple of 5
df = df[df['world_idx'] % 5 == 0]

# print(df[['world_idx', 'success']])
# only pick the first data of every world_idx
# df = df.groupby('world_idx').first().reset_index()

# clip by 0 to 0.5
# if not success, set score to 0
df["score"] = df["optimal_time"] / df["actual_time"] 
df["score"] = df["score"].clip(0.125, 0.5)
df.loc[df['success'] == False, 'score'] = 0
# print(df["score"])

print(df['world_idx'].value_counts().sort_index())
print(df['success'].value_counts())
print("Success rate: ", df['success'].value_counts()[0] / len(df))
print("Overall score: ", df['score'].mean())