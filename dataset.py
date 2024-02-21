import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

if __name__ == "__main__":
    print('Run main.py first!')

class DBS(Dataset):
    def __init__(self, dataframe, encoder, decoder, transform = None):
        super().__init__()
        self.df = dataframe
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
        self.img_dbs = pd.read_csv('data/Images.csv')
        
    def __getitem__(self, index):
        prod_record = self.df.iloc[index]
        prod_id = prod_record[0]
        #print(prod_id)
        n = self.img_dbs[self.img_dbs['product_id'] == prod_id]
        name_id = list(n['id'])
        #print(name_id)
        images = Image.new('RGB', (224, 224), color='white')
        for img in name_id:
            #images.append(Image.open(f'cleaned_images/{img}.jpg'))
            images = Image.open(f'cleaned_images/{img}.jpg')
            break
        #print(images)
        tensor = self.transform(images)
        prod_label = prod_record[-1]
        label = self.encoder[prod_label]
        #print(label)
        return (tensor, label)
    
    def __len__(self):
        return len(self.df)