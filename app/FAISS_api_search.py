import faiss
import numpy as np
import json
class Search:

    def __init__(self):

        with open('../image_embeddings.json', 'r') as json_file:
            self.img_embedding_dict = json.load(json_file)
        self.image_id_list = list(self.img_embedding_dict.keys())
        self.embeddings_list = np.array(list(self.img_embedding_dict.values()), dtype = np.float32)
        self.index = faiss.IndexFlatL2(self.embeddings_list.shape[1])
        self.index.add(self.embeddings_list)

    def search_img(self, image_id_emb):
        
        embeddings = np.array(image_id_emb.detach().numpy(), dtype=np.float32)
        k = 5
        similar_index = {}
        distances, neighbours = self.index.search(embeddings.reshape(1, -1), k)
        #print(f'Possible Nearest neighbours:')
        for i, neighbour_index in enumerate(neighbours.flatten()):
            neighbour_img_id = self.image_id_list[neighbour_index]
            distance = float(distances.flatten()[i])
            similar_index[neighbour_img_id] = distance
            #print(f'Neighbour {i+1}:\n Image ID: {neighbour_img_id}\n Distance: {distance}')
            
        return similar_index