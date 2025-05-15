import os 
import pandas as pd 
import numpy as np

class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class LibraryThing(DatasetLoader):
    def __init__(self, data_dir, ndatapoints):
        self.path = os.path.join(data_dir, 'reviews.txt')
        self.ndatapoints = ndatapoints

        
    def load(self):
        df = pd.DataFrame([], columns = ['comment','nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        file = open(self.path, 'r')
        lines = file.readlines()[1:]

        extracted_data = []
        linecount = 0

        for line in lines:
            try:
                line = line.split('] = ')
                line = line[1]
                reviews = eval(line)
                
                extracted_data.append([reviews.get('comment', ''), reviews.get('nhelpful', '0'), reviews.get('unixtime', '0'), reviews.get('work', ''), reviews.get('flags', ''), reviews.get('user', ''), reviews.get('stars', '0'), reviews.get('time', '')])
            except SyntaxError:
                pass

            if linecount > self.ndatapoints:
                    break
            linecount +=1

        

        #print(len(extracted_data))    
        

        df = pd.DataFrame(extracted_data, columns=['comment', 'nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        df['commentlength'] = df['comment'].str.split().apply(len)
        df.rename(columns = {'work':'item', 'stars':'rate'}, inplace = True)
        df['rate'] = df['rate'].astype(float)
        df['nhelpful'] = df['nhelpful'].astype(float)
        df['item'] = df['item'].astype(int)
        df['rate'] = [int(i) for i in np.ceil(df['rate'])]
        
        
        df, user_mapping = convert_unique_idx(df, 'user')
        df, item_mapping = convert_unique_idx(df, 'item')
     
        
        '''
        df['count'] = df.groupby('user')['user'].transform('size')
        df = df.sort_values('count', ascending=False)
        df = df[df['count'] > 20]
        df['count'] = df.groupby('item')['item'].transform('size')
        df = df.sort_values('count', ascending=False)
        df = df[df['count'] > 2]
        df = df[0:5000]
        print('number of unique users = ', len(df['user'].unique()))
        print('number of unique items = ',len(df['item'].unique()))
        print('number of data points = ', len(df))

        item_mapping = {}
        df.reset_index()
        print(df.head())
        print(df['key_item'].iloc[0])
        print(df['item'])
        key = df['key_item'].tolist()
        item = df['item'].tolist()
        for i in range(len(df)):
            item_mapping[key[i]] = item[i]
        print(len(item_mapping))
        '''
        
        


        return df, item_mapping

def convert_unique_idx(df, column_name):
    """ O: Switch the index notation of the movie reviews in DataFrame into an ordered system. 
    Return:
        df: Dataframe with new index system
        column_dict: Mapping between original index and new ordered index
    """
    # O: Build dictionary of index mappings from original definition to ordered definition
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    # O: Change index of dataframe
    df['key_' + column_name] = df[column_name]
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    
    # O: Check that new index system is of expected size
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1  

    return df, column_dict

    """

    def load(self):
        file = open(self.path, 'r')
        lines = file.readlines()

        extracted_data = []
        linecount = 0

        for line in lines:

            line_ = line.split('::')
            line_[0] = int(line_[0])
            line_[1] = int(line_[1])
            line_[2] = float(line_[2])
            line_[4] = float(line_[4])
            line_[5] = int(line_[5])
            line_[6] = int(line_[6])

            extracted_data.append(line_[:-1])

        df = pd.DataFrame(extracted_data, columns=['user', 'item', 'rate', 'unixtime', 'nhelpful', 'commentlength', 'key'])
        
        item_mapping = {}
        for i in range(len(df)):
            item_mapping[df['key'][i]] = df["user"][i]

        
            
        print('number of unique users = ', len(df['user'].unique()))
        print('number of unique items = ',len(df['item'].unique()))
        print('number of data points = ', len(df))
        


        return df, item_mapping
"""