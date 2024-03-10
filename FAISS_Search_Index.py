import faiss
import numpy as np
import json
class Search:

    def __init__(self):

        with open('image_embeddings.json', 'r') as json_file:
            self.img_embedding_dict = json.load(json_file)
        self.image_id_list = list(self.img_embedding_dict.keys())
        self.embeddings_list = list(self.img_embedding_dict.values())
        self.index = faiss.IndexFlatL2(1)
        self.index.add(self.embeddings_list)
        
    def save_func(self):
        faiss.write_index(self.index, 'appended_file.pkl')

    def search_img(self, image_id):

        if image_id not in self.img_embedding_dict:
            print(f"Image ID {image_id} not found in the dictionary.")
            return
        
        embeddings = np.array(self.img_embedding_dict[image_id], dtype=np.float32)
        k = 5
        similar_index = {}
        distances, neighbours = self.index.search(embeddings.reshape(1, -1), k)
        #print(f'Possible Nearest neighbours:')
        for i, neighbour_index in enumerate(neighbours.flatten()):
            neighbour_img_id = self.image_id_list[neighbour_index]
            distance = distances.flatten()[i]
            similar_index[neighbour_img_id] = distance
            #print(f'Neighbour {i+1}:\n Image ID: {neighbour_img_id}\n Distance: {distance}')
            
        return similar_index