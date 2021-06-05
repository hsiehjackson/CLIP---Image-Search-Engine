import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset.tsv',sep='\t')
train, test = train_test_split(df, test_size=0.1)
print(len(train), len(test))

with open('train.tsv', 'w') as f:
    train.to_csv(f, sep='\t', index=False)

with open('dev.tsv', 'w') as f:
    test.to_csv(f, sep='\t',index=False)