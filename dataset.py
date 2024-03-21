import pandas as pd # Import Essential Packages and Classes
from torch.utils.data import Dataset
import torch

# Check if this script is being run directly
if __name__ == "__main__":
    print('Run main.py first!')

class DBS(Dataset):

    def __init__(self, dataframe, encoder, decoder, transform = None):

        """
        Initialize the dataset with provided data and mappings.
        dataframe: The DataFrame containing product data
        encoder: A dictionary mapping product labels to encoded values
        decoder: A dictionary mapping encoded values back to product labels
        transform: Optional transform to be applied to the data
        """

        super().__init__()
        self.df = dataframe # Assign the given values
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transform
        self.img_dbs = pd.read_csv('data/Images.csv') # Loading image database
        
    def __getitem__(self, index):

        """
        Retrieve an item from the dataset.
        index: Index of the item to retrieve
        """

        prod_record = self.df.iloc[index] # Get product record at given index
        prod_id = prod_record[0] # Extract product ID
        n = self.img_dbs[self.img_dbs['product_id'] == prod_id] # Filter image database for product ID
        name_id = list(n['id']) # Extract image IDs associated with the product
        image = '' # Placeholder for image path

        for img_path in name_id:
            image = f'cleaned_images/{img_path}.jpg' # Construct image path
            break # Break after finding the first image associated with the product

        img_tensor = [] # Placeholder for image tensor
        
        if image == '':
            prod_label = 'white-image' # If no image found, label it as 'white-image'
        else:
            prod_label = prod_record[-1] # Extract product label
        
        img_tensor.append(image) # Append image path to the list
        label = torch.tensor(self.encoder[prod_label], dtype=torch.long) # Encode label
        return (img_tensor, label) # Return tuple containing image tensor and label tensor
    
    def __len__(self):

        """
        Get the length of the dataset
        """

        return len(self.df)