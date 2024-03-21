# Project Title: Facebook Marketplace's Recommendation Ranking System

>## Project Description:
This project implements a recommendation system inspired by Facebook Marketplace, leveraging artificial intelligence to recommend relevant listings based on personalized search queries.

>## Table of Contents:

1. [Installation Instructions](#installation-instructions)
2. [Methodology](#methodology)
    - [1. Image Processing (clean_images.py)](#1-image-processing-clean_imagespy)
    - [2. Tabular Data Cleaning (clean_tabular_data.py)](#2-tabular-data-cleaning-clean_tabular_datapy)
    - [3. Dataset Creation (dataset.py)](#3-dataset-creation-datasetpy)
    - [4. FAISS Search Index Creation (FAISS_Search_Index.py)](#4-faiss-search-index-creation-faiss_search_indexpy)
    - [5. Model Training (pretrain_load.py)](#5-model-training-pretrain_loadpy)
    - [6. Main Execution (main.py)](#6-main-execution-mainpy)
3. [File Structure](#file-structure)
4. [License Information](#license-information)

>## Installation Instructions:
- Clone the repository:
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository

>## Installation Instructions:
- git clone https://github.com/jai15111997/multinational-retail-data-centralisation228
- Request for db_credentials.yaml file from AiCore to gain access to the database.
- Add local server credentials to db_credentials.yaml as following:
    - HOST: {Host Name}
    - USER: {Username}
    - PASSWORD: {Set Password}
    - DATABASE: sales_data
    - PORT: {Port Number}
- Run main.py file

>## Methodology

Methodology

***1. Image Processing (clean_images.py):***
Resize and process images for model training.

***2. Tabular Data Cleaning (clean_tabular_data.py):***
Clean and preprocess tabular data for model training.

***3. Dataset Creation (dataset.py):***
Create datasets for training, validation, and testing.

***4. FAISS Search Index Creation (FAISS_Search_Index.py):***
Create a FAISS search index for fast similarity search.

***5. Model Training (pretrain_load.py):***
Train a model using preprocessed data.

***6. Main Execution (main.py):***
Orchestrates the entire pipeline including data preprocessing, model training, and search index creation.

>## File Structure:

    - README.md: Overview of the project's code and methods.
    - clean_images.py: Script for image processing.
    - clean_tabular_data.py: Script for tabular data cleaning.
    - dataset.py: Script for dataset creation.
    - FAISS_Search_Index.py: Script for FAISS search index creation.
    - pretrain_load.py: Script for model training.
    - main.py: Main execution script.


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