import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

if __name__ == "__main__":
    print('Run main.py first!')

class DBS(Dataset):
    def __init__(self, dataframe, encoder, decoder, transform = None):
        super().__init__()
        self.df = dataframe
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transform
        self.img_dbs = pd.read_csv('data/Images.csv')
        
    def __getitem__(self, index):
        prod_record = self.df.iloc[index]
        prod_id = prod_record[0]
        #print(prod_id)
        n = self.img_dbs[self.img_dbs['product_id'] == prod_id]
        name_id = list(n['id'])
        #print(name_id)
        image = ''
        for img_path in name_id:
            image = f'cleaned_images/{img_path}.jpg'
            break
        img_tensor = []
        img_tensor.append(image)
        prod_label = prod_record[-1]
        label = torch.tensor(self.encoder[prod_label], dtype=torch.long)
        return (img_tensor, label)
    
    def __len__(self):
        return len(self.df)