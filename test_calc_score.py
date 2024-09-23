import pandas as pd

df = pd.read_csv('imit_data.csv', sep=' ')

score = 0
for i in range(df.shape[0]):
    score += df.iloc[i]['score']
print(score / df.shape[0])