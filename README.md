# Project Title: Facebook Marketplace's Recommendation Ranking System

>## Project Description:
Facebook Marketplace is a platform for buying and selling products on Facebook.

This is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

>## Table of Contents:

1. [Installation Instructions](#installation-instructions)
2. [Methodology](#methodology)
    - [1. Extracting User Details from Amazon RDS](#1-extracting-user-details-from-amazon-rds)
    - [2. Retrieving User Card Details from AWS S3](#2-retrieving-user-card-details-from-aws-s3)
    - [3. Storing Data through API Utilization](#3-storing-data-through-api-utilization)
    - [4. Extracting Product Details from AWS S3](#4-extracting-product-details-from-aws-s3)
    - [5. Extracting Order Details from Amazon RDS](#5-extracting-order-details-from-amazon-rds)
    - [6. Data Cleaning](#6-data-cleaning)
    - [7. Data Loading](#7-data-loading)
    - [8. Data Analysis](#8-data-analysis)
3. [Usage Instructions](#usage-instructions)
    - [Data Extraction (data_extraction.py)](#data-extraction-data_extractionpy)
    - [Data Utilities (database_utils.py)](#data-utilities-database_utilspython)
    - [Data Cleaning (data_cleaning.py)](#data-cleaning-data_cleaningpy)
    - [SQL Operations (SQL_Operations.py)](#sql-operations-sql_operationspy)
    - [SQL Queries (SQL_Queries.py)](#sql-queries-sql_queriespy)
    - [Main (main.py)](#main-mainpy)
4. [File Structure](#file-structure)
5. [License Information](#license-information)

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

***1. Extracting User Details from Amazon RDS:***
The archival data of users is presently housed in an AWS cloud-based database. The DatabaseConnector class within the database_utils.py script facilitates the connection to the Amazon RDS database. In the DataExtractor class within data_extraction.py, a method is devised to extract the user details table and transform it into a Pandas DataFrame.

***2. Retrieving User Card Details from AWS S3:***
The details of users' payment cards are stored in a PDF document within an AWS S3 bucket. The installation of the Python package tabula-py proves instrumental in extracting data from the PDF document. A method is established in the DataExtractor class, accepting a link as an argument and returning a Pandas DataFrame.

***3. Storing Data through API Utilization:***
The API comprises two GET methods—one for providing the count of stores in the business and the other for fetching details about a specific store. To connect to the API, the method header needs to incorporate the API key. A header dictionary is created with the 'x-api-key' as the key and its corresponding value. The API endpoints are designed for retrieving a store and returning the total number of stores. A DataExtractor method is developed to determine the count of stores by utilising the number of stores endpoint and the header dictionary. Another method is crafted to retrieve details about a store using the corresponding endpoint, saving the data into a Pandas DataFrame.

***4. Extracting Product Details from AWS S3:***
Information pertaining to each product the company currently offers is stored in CSV format within an S3 bucket on AWS. A DataExtractor method is created, leveraging the boto3 package to download and extract the product information, ultimately returning a Pandas DataFrame. The S3 address for the products data (s3://data-handling-public/products.csv) is passed as an argument to this method.

***5. Extracting Order Details from Amazon RDS:***
The table serving as the definitive record for all past orders made by the company is stored in a database on AWS RDS. A method is formulated to extract the order data table and transform it into a Pandas DataFrame.

***6. Data Cleaning:***
To ensure data integrity, several checks and corrections are applied to each dataframe:

- Identification and removal of duplicates.
- Correction of column values with incorrect data types.
- Validation of date entries for errors.
- Conversion of columns to appropriate data types.
- Elimination of rows containing erroneous or NULL values.
- Removal of columns lacking meaningful data.

***7. Data Loading:***
Upon completing the data cleaning process, the dataframes are stored as tables in the pgAdmin 4 sales_data database. SQL queries are then utilised for interacting with these tables.

- User details table: dim_users
- User card details table: dim_card_details
- Store details table: dim_store_details
- Product details table: dim_products
- Order details table: orders_table
- All tables contribute to the orders_table, serving as the comprehensive source for all order-related information.

With primary keys prefixed by "dim" in the tables, foreign keys are established in the orders_table to reference primary keys in other tables.

Through the application of SQL, foreign key constraints are implemented, referencing the primary keys of other tables and finalising the star-based database schema.

***8. Data Analysis:***
Now that the data has been properly organised, several SQL queries are run to extract up-to-date metrics from the data. This initiative aims to empower the concerned business to made more informed, data-driven decisions and to enhance its comprehension of sales dynamics. In this milestone, the following business queries have been responded to via pgAdmin4 and python:

- How many stores does the business have and in which countries?
- Which locations currently have the most stores?
- Which months produced the largest amount of sales?
- How many sales are coming from online?
- What percentage of sales come through each type of store?
- Which month in each year produced the highest cost of sales?
- What is our staff headcount?
- Which German store type is selling the most?
- How quickly is the company making sales?

>## Usage Instructions:

### Data Extraction (data_extraction.py):
**DataExtractor Class:**

        - retrieve_pdf_data(pdf_path): Retrieves data from a PDF file.
        - list_db_tables(): Lists tables in the database.
        - read_rds_table(db_connector, table_name): Reads a table from an RDS database.
        - list_number_of_stores(api_endpoint, headers): Retrieves the number of stores from an API endpoint.

### Data Utilities (database_utils.py):
**DatabaseConnector Class:**

        - read_db_creds(): Reads database credentials.
        - upload_to_db(data_frame, table_name): Uploads data to the specified table in the database.

### Data Cleaning (data_cleaning.py):
**DataCleaning Class:**

        - clean_card_data(data_frame): Cleans and refines card data.
        - clean_user_data(data_frame): Cleans and refines user data.
        - clean_orders_data(data_frame): Cleans and refines order data.
        - called_clean_store_data(data_frame): Cleans and refines store data.
        - convert_product_weights(data_frame): Converts product weights.

### SQL Operations (SQL_Operations.py):
**SQL_datatype_change Class:**

        - dtype_change(): Performs SQL operations to change data types and structure.

### SQL Queries (SQL_Queries.py):
**SQL_queries Class:**

        - QnA(): Executes SQL queries to answer specific business questions.

### Main (main.py):
**Main Execution Script:**

        - Initiates the entire data processing pipeline.
        - Connects to the database.
        - Extracts data from various sources.
        - Cleans and uploads data to the database.
        - Executes SQL operations and queries.

>## File Structure:
    - README.md: Overview of the project's code and methods.
    - data_extraction.py: Data extraction functions.
    - database_utils.py: Database utility functions.
    - data_cleaning.py: Data cleaning functions.
    - SQL_Operations.py: SQL operations functions.
    - SQL_Queries.py: SQL query functions.
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