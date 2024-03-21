from clean_images import cleaning_images # Importing Essential classes and Packages
from clean_tabular_data import prod_clean
from dataset import DBS
from FAISS_Search_Index import Search
from pretrain_load import Pretrained
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

clean_tabular_data = prod_clean() # Creating instances of Classes
resize_img_exec = cleaning_images()

df = prod_clean.data_clean() # Cleaning Image data into Dataframe

df['labels'] = df['category'].apply(lambda x: x.split('/')[0]) # Extract labels from category
label_list = df['labels'].unique() # Get unique labels

encoder = {} # Initialize label encoder dictionary
decoder = {} # Initialize label decoder dictionary

for x in range(0, len(label_list)):

    # Create encoder and decoder dictionaries
    decoder[x] = label_list[x]
    value = label_list[x]
    encoder[value] = x

# Add 'white-image' label to encoder and decoder in case of no-match in the Records
encoder['white-image'] = 13 
decoder[13] = 'white-image'

df.to_csv('data/training_data.csv',) # Save cleaned data to CSV file

resize_img_exec.process_images() # Process and resize images

# Split data into training, validation, and test sets
train_df, rest_df = train_test_split(df, test_size = 0.3, random_state = 42)
validation_df, test_df = train_test_split(rest_df, test_size = 0.5, random_state = 42)

# Create datasets and dataloaders
train_dataset = DBS(train_df, encoder, decoder, transform = None)
validation_dataset = DBS(validation_df, encoder, decoder, transform = None)
test_dataset = DBS(test_df, encoder, decoder, transform = None)

# Create Dataloaders for each Dataset
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = 8, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

# Train the model
p_train = Pretrained(train_dataset, train_dataloader)
p_train.train(train_dataloader, validation_dataloader, 8) # 8 Epochs

# Save the Trained Model Weights
torch.save(p_train.state_dict(), 'final_model/image_model.pt')

# Create and save FAISS search index
query = Search()
query.save_func()