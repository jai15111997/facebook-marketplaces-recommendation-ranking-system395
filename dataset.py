import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

if __name__ == "__main__":
    print('Run main.py first!')

class DBS(Dataset):
    def __init__(self, dataframe, encoder, decoder, transform = None):
        super().__init__()
        self.df = dataframe
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256), transforms.RandomHorizontalFlip(p=0.3), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.img_dbs = pd.read_csv('data/Images.csv')
        
    def __getitem__(self, index):
        prod_record = self.df.iloc[index]
        prod_id = prod_record[0]
        #print(prod_id)
        n = self.img_dbs[self.img_dbs['product_id'] == prod_id]
        name_id = list(n['id'])
        #print(name_id)
        image = Image.new('RGB', (256, 256), color='white')
        for img_path in name_id:
            image = Image.open(f'cleaned_images/{img_path}.jpg')
            break
        img_tensor = self.transform(image)
        prod_label = prod_record[-1]
        label = torch.tensor(self.encoder[prod_label], dtype=torch.long)
        return (img_tensor, label)
    
    def __len__(self):
        return len(self.df)