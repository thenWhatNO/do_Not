import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

data_path = "data_2/object_label_cords.csv"

df = pd.read_csv(data_path)
images= df['image']
index = df['ind']
top_l = df['top_L']
top_l2 = df['top_L2']
down_r = df['dotom_R']
down_r2 = df['dotom_R2']

print(df.head())