import numpy as np
import pandas as pd

data_path = "tokens.csv"

text = "we get a new fishes in our garden"

df = pd.read_csv(data_path)
work = df['word']
id  = df['id']
toke = df['token']

embedding_dim = 4

text += ' '

word = ''

for letter in text:
    word += letter
    if letter == ' ':
        word = word[:-1]

        if not (work == word).any():
            token = np.random.randn(embedding_dim)

            df.loc[len(df)] = [word, id.iloc[-1]+1, token]

        word = ''
        work = df['word']
        id  = df['id']
        toke = df['token']

df.to_csv('tokens.csv', index=False)