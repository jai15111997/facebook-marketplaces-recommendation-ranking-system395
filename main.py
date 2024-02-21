import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from clean_tabular_data import prod_clean
from clean_images import cleaning_images
from dataset import DBS
from pretrain_load import Pretrained

clean_tabular_data = prod_clean()
resize_img_exec = cleaning_images()
df = prod_clean.data_clean()
df['labels'] = df['category'].apply(lambda x: x.split('/')[0])
label_list = df['labels'].unique()
encoder = {}
decoder = {} 
for x in range(0, len(label_list)):
    decoder[x] = label_list[x]
    value = label_list[x]
    encoder[value] = x
#df.to_csv('data/training_data.csv',)

#resize_img_exec.process_images()

train_df, rest_df = train_test_split(df, test_size = 0.3, random_state = 42)
validation_df, test_df = train_test_split(rest_df, test_size = 0.5, random_state = 42)

train_dataset = DBS(train_df, encoder, decoder, transform = None)
validation_dataset = DBS(validation_df, encoder, decoder, transform = None)
test_dataset = DBS(test_df, encoder, decoder, transform = None)

train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = 8, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

p_train = Pretrained(train_dataset, train_dataloader)
p_train.train(train_dataloader, validation_dataloader, 2)

