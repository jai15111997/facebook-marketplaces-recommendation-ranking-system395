import pandas as pd # Importing the pandas library for data manipulation

# Check if this script is being run directly
if __name__ == "__main__":
    print('Run main.py first!')
    
class prod_clean:       

    def data_clean():

        """
        Perform data cleaning on the Products.csv file
        """

        df = pd.read_csv('data/Products.csv', lineterminator='\n') # Read the CSV file into a DataFrame, specifying line terminator to avoid parsing issues
        df.drop('Unnamed: 0', axis = 1, inplace = True) # Drop the 'Unnamed: 0' column
        df.dropna(inplace = True) # Drop rows with missing values

        # Convert specific columns to string data type
        df['id'] = df['id'].astype('string')
        df['product_name'] = df['product_name'].astype('string')
        df['category'] = df['category'].astype('string')
        df['product_description'] = df['product_description'].astype('string')
        df['location'] = df['location'].astype('string')

        # Clean the 'price' column by removing currency symbols and commas, then convert to numeric
        df['price'] = df['price'].apply(lambda x: x.strip('£').strip(',') if ('£' in x or ',' in x) else x)
        df['price'] = pd.to_numeric(df['price'], errors = 'coerce')
        df.dropna(inplace = True) # Drop rows with missing or invalid 'price'
        return df