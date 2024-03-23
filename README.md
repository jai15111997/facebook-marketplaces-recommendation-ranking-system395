# Project Title: Facebook Marketplace's Recommendation Ranking System

>## Project Description:
Facebook Marketplace is a popular platform for buying and selling a wide range of products within local communities. Leveraging the success of such platforms, this project, developed by AiCore, aims to implement a recommendation system inspired by Facebook Marketplace. The system utilizes artificial intelligence techniques to enhance the user experience by recommending relevant listings based on personalized search queries.

#### &emsp; *Key Features*:

- ***Personalized Recommendations:*** The recommendation system employs advanced algorithms to analyze user preferences and behaviors, providing personalized recommendations tailored to each user's interests and needs.

- ***AI-powered Search:*** Leveraging artificial intelligence techniques such as Natural Language Processing (NLP) and computer vision, the system enhances search capabilities by understanding user queries and extracting relevant information from textual descriptions and images of listings.

- ***Content-based Filtering:*** The recommendation system employs content-based filtering techniques to suggest listings similar to those previously viewed or interacted with by the user. This approach ensures that recommendations align closely with the user's preferences and interests.

- ***Collaborative Filtering:*** In addition to content-based filtering, the system incorporates collaborative filtering methods to identify patterns and similarities among users, allowing for the generation of recommendations based on the preferences and behaviors of similar users.

- ***Real-time Updates:*** The recommendation system continuously updates its recommendations based on user interactions and feedback, ensuring that the suggestions remain relevant and up-to-date.

- ***Scalability and Performance:*** Built with scalability and performance in mind, the system can efficiently handle large volumes of data and user interactions, providing seamless and responsive recommendations even under heavy load conditions.

#### &emsp; *Use Cases*:
- ***Personal Shopping Assistant:*** Users can benefit from personalized recommendations tailored to their specific interests and preferences, making the shopping experience more enjoyable and efficient.

- ***Improved User Engagement:*** By providing relevant and timely recommendations, the system encourages users to explore more listings and engage actively with the platform, leading to increased user satisfaction and retention.

- ***Enhanced Product Discovery:*** The recommendation system helps users discover new and relevant products that they may not have found through traditional search methods, expanding their choices and enhancing their shopping experience.

- ***Increased Marketplace Revenue:*** By promoting listings through targeted recommendations, sellers can reach a broader audience of potential buyers, leading to increased visibility and sales on the platform.

&emsp;&emsp;This project aims to revolutionize the online marketplace experience by harnessing the power of artificial intelligence to deliver personalized and engaging recommendations to users, ultimately driving greater user satisfaction, engagement, and business success.

>## Table of Contents:

1. [Installation Instructions](#installation-instructions)
2. [Methodology](#methodology)
    - [1. Image Resizing (clean_images.py)](#1-image-resizing-clean_imagespy)
    - [2. Tabular Data Cleaning (clean_tabular_data.py)](#2-tabular-data-cleaning-clean_tabular_datapy)
    - [3. Dataset Creation (dataset.py)](#3-dataset-creation-datasetpy)
    - [4. FAISS Search Index Creation (FAISS_Search_Index.py)](#4-faiss-search-index-creation-faiss_search_indexpy)
    - [5. Image Processing (image_processor.py)](#5-image-processing-image_processorpy)
    - [6. Model Training (pretrain_load.py)](#6-model-training-pretrain_loadpy)
    - [7. Main Execution (main.py)](#7-main-execution-mainpy)
    - [8. Image Processing Utility (api_image_processing.py)](#8-image-processing-utility-api_image_processingpy)
    - [9. FAISS Search API (FAISS_api_search.py)](#9-faiss-search-api-faiss_api_searchpy)
    - [10. Image Processing API (api.py)](#10-image-processing-api-apipy)
3. [Usage Instructions](#usage-instructions)
    - [1. api.py]()
    - [2. api_image_processor.py]()
    - [3. clean_images.py]()
    - [4. clean_tabular_data.py]()
    - [5. dataset.py]()
    - [6. FAISS_api_search.py]()
    - [7. FAISS_Search_Index.py]()
    - [8. image_processor.py]()
    - [9. pretrain_load.py]()
    - [10. main.py]()
4. [File Structure](#file-structure)
5. [License Information](#license-information)

>## Installation Instructions:
- Git clone https://github.com/jai15111997/facebook-marketplaces-recommendation-ranking-system395 to a local repository
- Request for AWS login credentials from AiCore to gain access to the S3 Bucket.
- Download the EC2 'pem' file to access the EC2 instance from the local terminal.
- Access the EC2 instance and download 'Images.csv', 'images_fb.zip' and 'Products.csv' files to the local repository.
- Extract the images zip file.
- Create a new folder inside the same repository by the name 'data' and transfer both csv files to it.
- Run main.py file and wait for it to perform all the operations.
- Copy 'appended_file.pkl' and 'image_embeddings.json' to 'app' folder.
- Copy 'image_model.pt' from 'final_model' folder to 'app' folder.
- Run 'api.py' to deploy API and run it form a browser by accessing: http://localhost:8080/docs
- To make a docker image for the same, use the docker build command in the 'app' folder from the terminal.

>## Methodology

***1. Image Resizing (clean_images.py)***:
Resize images for model training for faster image processing.

***2. Tabular Data Cleaning (clean_tabular_data.py):***
Clean and preprocess tabular data for model training.

***3. Dataset Creation (dataset.py):***
- Create datasets for training, validation, and testing.
- Encode labels numerically using encoders and decoders to prepare them for model predictions.

***4. FAISS Search Index Creation (FAISS_Search_Index.py):***
Create a FAISS search index for fast similarity search.

***5. Image Processing (image_processor.py):***
- Process batches of images and convert them into tensors.
- Stack all transformed images along a specified dimension and record the image name and its converted tensor as a dictionary for each batch passed.

***6. Model Training (pretrain_load.py):***
- Train a model using preprocessed data, utilizing the ResNet-50 model architecture.
- Record model performance metrics and accuracy with each timestamp using TensorBoard.
- Record the image IDs and corresponding tensors as a dictionary for search purposes.

***7. Main Execution (main.py):***
- Orchestrates the entire pipeline, including data preprocessing, model training, and search index creation.
- Manages the flow of data and execution of each step in the pipeline to ensure a smooth and efficient process.

***8. Image Processing Utility (api_image_processing.py):***
- Image Transformation: Define a class for image utility functions, including methods for transforming images into tensors suitable for model input.
- Batch Image Processing: Implement a function to process batches of images and convert them into tensors, stacking them along a specified dimension.

***9. FAISS Search API (FAISS_api_search.py):***
- Search Index Initialization: Initialize a FAISS search index using pre-computed image embeddings stored in a JSON file.
- Image Similarity Search: Create a method to perform similarity search on the FAISS index using feature embeddings of uploaded images.

***10. Image Processing API (api.py):***
- FastAPI Integration: Utilize the FastAPI framework to develop a RESTful API for image processing tasks.
- Health Check Endpoint: Implement a health check endpoint to ensure the API is up and running.
- Feature Embedding Prediction: Create an endpoint to predict feature embeddings of uploaded images using a pre-trained feature extraction model.
- Similar Images Prediction: Develop an endpoint to predict similar images based on feature embeddings using a pre-built FAISS search index.

>## Usage Instructions:

### &emsp;1. API Execution (api.py):

**FeatureExtractor Class:**

    - __init__(decoder): Initializes the class with a decoder dictionary and defines the ResNet50 model with additional layers.
    - forward(image): Forward pass of the model, taking an image tensor as input and returning the output tensor.
    - predict(image): Makes predictions using the model on the input image tensor.

**Endpoints:**

    - /healthcheck: GET request endpoint to verify server status.
    - /predict/feature_embedding: POST request endpoint to predict image embeddings. Accepts an image file and returns the image embeddings as JSON response.
    - /predict/similar_images: POST request endpoint to predict similar images using FAISS index. Accepts an image file and returns the index of similar images as JSON response.

### &emsp;2. Image Processing Utility (api_image_processor.py):

**image_utility Class:**

    - __init__(): Initializes the image processor utility with image transformation settings.
    - process_image(image_batch): Processes a batch of images and converts them into tensors.

### &emsp;3. Image Resizing (clean_images.py):

**cleaning_images Class:**

    - resize_image(final_size, im): Resizes images to a specified final size.
    - process_images(): Processes images in a directory by resizing them and saving the resized versions.

### &emsp;4. Tabular Data Cleaning (clean_tabular_data.py):

**prod_clean Class:**

    - data_clean(): Cleans and preprocesses tabular data for model training.

### &emsp;5. Dataset Creation (dataset.py):

**DBS Class:**

    - __init__(dataframe, encoder, decoder, transform): Initializes the dataset with provided data, encoders, decoders, and transformation settings.
    - __getitem__(index): Retrieves an item from the dataset based on the provided index.
    - __len__(): Returns the length of the dataset.

### &emsp;6. FAISS Search API (FAISS_api_search.py):

**Search Class:**

    -__init__(): Initializes the FAISS search index with image embeddings.
    - search_img(image_id_emb): Searches for similar images based on the provided image embeddings.

### &emsp;7. FAISS Search Index Creation (FAISS_Search_Index.py):

**Search Class:**

    - __init__(): Initializes the class by loading image embeddings from a JSON file, extracting image IDs and embeddings, creating a FAISS index for L2 distance, and adding the embeddings to the index.
    - save_func(): Saves the created FAISS index to a file for future use.

### &emsp;8. Image Processing (image_processor.py):
**image_utility Class:**
    - __init__(): Initializes the image utility with image transformation settings.
    - process_image(image_batch): Processes a batch of images and converts them into tensors.
    - dict_updater(image_batch, predictions): Updates a dictionary with image predictions.

### &emsp;9. Model Training (pretrain_load.py):

**Pretrained Class:**

    - __init__(dataset, dataloader): Initializes the model for training with the provided dataset and dataloader.
    - forward(inp): Forward pass through the model.
    - save_checkpoint(epoch, folder_path): Saves the model checkpoint at the specified epoch.
    - train(dataloader, validation_dl, epochs): Trains the model using the provided dataloader and validation data for the specified number of epochs. Saves image embeddings to JSON file at the end.

### &emsp;10. Main Execution (main.py):

**Main Script Execution:**

    - Initiates the entire pipeline including data preprocessing, model training, and search index creation.

>## File Structure:

    - README.md: Overview of the project's code and methods.
    - api.py: Script for implementing the image processing API using FastAPI.
    - api_image_processor.py: Script containing utility functions for image processing in the API.
    - FAISS_api_search.py: Script for creating and utilizing a FAISS search index in the API.
    - clean_images.py: Script for resizing and processing images.
    - clean_tabular_data.py: Script for cleaning and preprocessing tabular data.
    - dataset.py: Script for creating datasets for model training.
    - FAISS_Search_Index.py: Script for creating a FAISS search index for fast similarity search.
    - image_processor.py: Script for processing images and converting them into tensors.
    - pretrain_load.py: Script for training a model using preprocessed data.
    - main.py: Main execution script orchestrating the entire pipeline.
    - dockerfile: Dockerfile for containerizing the application environment.
    - requirements.txt: File specifying the dependencies required for running the application.
    - appended_file.pkl: Serialized FAISS search index file.
    - image_model.pt: Pre-trained model weights file for image feature extraction.
    - image_embeddings.json: JSON file containing image embeddings for FAISS search index.
    - Images.csv: CSV file containing image data.
    - Products.csv: CSV file containing product data.
    - training_data.csv: CSV file containing training data for the model.


>## License information:

MIT License

Copyright (c) [2024] [AiCore]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.