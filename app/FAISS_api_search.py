import faiss # Import Essential Packages
import json
import numpy as np

# Check if this script is being run directly
if __name__ == "__main__":
    print('api.py first!')

class Search:

    """
    Class for performing similarity search using FAISS index.
    """

    def __init__(self):
        
        # Load image embeddings from JSON file
        with open('image_embeddings.json', 'r') as json_file:
            self.img_embedding_dict = json.load(json_file)

        # Extract image IDs and embeddings    
        self.image_id_list = list(self.img_embedding_dict.keys())
        self.embeddings_list = np.array(list(self.img_embedding_dict.values()), dtype = np.float32)
        
        # Initialize FAISS index and add embeddings to the index
        self.index = faiss.IndexFlatL2(self.embeddings_list.shape[1]) # Create a flat L2 index
        self.index.add(self.embeddings_list) # Add embeddings to the index

    def search_img(self, image_id_emb):
        
        """
        Method to perform similarity search for a given image embedding.

        image_id_emb: Image embedding tensor.

        similar_index: Dictionary containing similar image IDs and their distances.
        """

        # Convert image embedding tensor to NumPy array
        embeddings = np.array(image_id_emb.detach().numpy(), dtype=np.float32)
        
        # Define the number of nearest neighbors to search for
        k = 5

        # Perform similarity search and retrieve nearest neighbors
        similar_index = {}
        distances, neighbours = self.index.search(embeddings.reshape(1, -1), k)
        
        # Process the nearest neighbors and their distances
        for i, neighbour_index in enumerate(neighbours.flatten()):
            
            neighbour_img_id = self.image_id_list[neighbour_index]
            distance = float(distances.flatten()[i])
            similar_index[neighbour_img_id] = distance
            
        return similar_index # Return dictionary of similar image IDs and distances