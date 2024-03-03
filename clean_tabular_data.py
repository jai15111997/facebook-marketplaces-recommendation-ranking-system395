import pandas as pd

if __name__ == "__main__":
    print('Run main.py first!')
    
class prod_clean:       
    def data_clean():
        df = pd.read_csv('data/Products.csv', lineterminator='\n')
        df.drop('Unnamed: 0', axis = 1, inplace = True)
        df.dropna(inplace = True)
        df['id'] = df['id'].astype('string')
        df['product_name'] = df['product_name'].astype('string')
        df['category'] = df['category'].astype('string')
        df['product_description'] = df['product_description'].astype('string')
        df['location'] = df['location'].astype('string')
        df['price'] = df['price'].apply(lambda x: x.strip('£').strip(',') if ('£' in x or ',' in x) else x)
        df['price'] = pd.to_numeric(df['price'], errors = 'coerce')
        df.dropna(inplace = True)
        return df