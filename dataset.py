import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

if __name__ == "__main__":
    print('Run main.py first!')

class DBS(Dataset):
    def __init__(self, dataframe, encoder, decoder, transform = None):
        super().__init__()
        self.df = dataframe
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transform
        
    def __getitem__(self, index):
        name = self.df['id']
        img = Image.open(f'cleaned_images/{name}_resized.jpg')
        y = self.encoder[self.df['labels'][index]]
        return (img, y)
    
    def __len__(self):
        return len(self.df)