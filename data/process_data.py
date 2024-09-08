import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Input:
    messages_filepath: Path to the messages CSV file.
    categories_filepath: Path to the categories CSV file.

    Output:
    pd.DataFrame: Merged dataframe containing messages and categories.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
   
    return df

def clean_data(df):
    """
    Clean the dataframe by splitting the 'categories' column
    into separate category columns and converting category values 
    to numerical values.

    input:
    df: Dataframe containing the 'categories' column.

    output:
    DataFrame: Cleaned dataframe with individual category columns.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    # rename the columns of `categories`
    row = categories.iloc[0, :]
    category_colnames = row.str[:-2]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # Remove case multiclass in "related" column
    df = df[df['related'] != 2]

    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database.

    Input:
    df: Dataframe to be saved to the database.
    database_filename: Path to the SQLite database file.
    """
    conn = sqlite3.connect(database_filename)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS InsertTableName")
    conn.close()
    
    # Load data into InsertTableName
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    """
    Main function to load, clean, and save data based on command-line arguments.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()