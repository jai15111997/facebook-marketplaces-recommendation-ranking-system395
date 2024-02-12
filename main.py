import pandas as pd
import numpy as np

from clean_tabular_data import prod_clean
from clean_images import cleaning_images

clean_tabular_data = prod_clean()
resize_img_exec = cleaning_images()
df = prod_clean.data_clean()
df['labels'] = df['category'].apply(lambda x: x.split('/')[0])
label_list = df['labels'].unique()
encoder = {}
decoder = {} 
for x in range(0, len(label_list)):
    encoder[x] = label_list[x]
    value = label_list[x]
    decoder[value] = x
df.to_csv('data/training_data.csv')

resize_img_exec.process_images()