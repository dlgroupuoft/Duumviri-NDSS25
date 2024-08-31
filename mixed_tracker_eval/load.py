import code
import pandas as pd


# read tsv file
df_p = pd.read_csv('positive.tsv', sep='\t')
df_n = pd.read_csv('negative.tsv', sep='\t')


print("--------- positive samples -----------")
print(df_p.head(3))
print(df_p.columns)
print(df_p.shape)
print(df_p['Manual Label'].value_counts())

print("--------- negative samples -----------")
print(df_n.head(3))
print(df_n.columns)
print(df_n.shape)
print(df_n['Manual Label'].value_counts())

code.interact(local=dict(globals(), **locals()))
