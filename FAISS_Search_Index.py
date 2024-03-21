import faiss # Importing the Faiss library for similarity search
import json # Importing the json module for JSON manipulation
import numpy as np # Importing the NumPy library for numerical computations

# Check if this script is being run directly
if __name__ == "__main__":
    print('Run main.py first!')
    
class Search:

    def __init__(self):

        # Load image embeddings from JSON file
        with open('image_embeddings.json', 'r') as json_file: 
            self.img_embedding_dict = json.load(json_file)

        # Extract image IDs and embeddings
        self.image_id_list = list(self.img_embedding_dict.keys())
        self.embeddings_list = np.array(list(self.img_embedding_dict.values()), dtype = np.float32)
        
        # Create a flat index for L2 distance
        self.index = faiss.IndexFlatL2(self.embeddings_list.shape[1])
        self.index.add(self.embeddings_list)
        
    def save_func(self):
        faiss.write_index(self.index, 'appended_file.pkl') # Save the index to a file