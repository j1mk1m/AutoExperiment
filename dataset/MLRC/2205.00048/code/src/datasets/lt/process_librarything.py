import pandas as pd 
import numpy as np
import os 

class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class LibraryThing(DatasetLoader):
    def __init__(self, data_dir, ndatapoints, idx_1, idx_2):
        self.path = os.path.join(data_dir, 'reviews.txt')
        self.ndatapoints = ndatapoints
        self.idx_1 = idx_1
        self.idx_2 = idx_2
    
    def load(self):
        df = pd.DataFrame([], columns = ['comment','nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        file = open(self.path, 'r')
        lines = file.readlines()[self.idx_1:self.idx_2]

        extracted_data = []
        linecount = 0
        for line in lines:
            try:
                line = line.split('] = ')
                line = line[1]
                reviews = eval(line)
                linecount +=1
                extracted_data.append([reviews.get('comment', ''), reviews.get('nhelpful', '0'), reviews.get('unixtime', '0'), reviews.get('work', ''), reviews.get('flags', ''), reviews.get('user', ''), reviews.get('stars', '0'), reviews.get('time', '')])
                if linecount%10000 == 0:
                    print(linecount/self.ndatapoints)
                if linecount > self.ndatapoints - 1:
                        break
            except SyntaxError:
                pass

        df = pd.DataFrame(extracted_data, columns=['comment', 'nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        df['commentlength'] = df['comment'].str.split().apply(len)
        df.rename(columns = {'work':'item', 'stars':'rate'}, inplace = True)
        df['rate'] = df['rate'].astype(float)
        df['nhelpful'] = df['nhelpful'].astype(float)
        df['item'] = df['item'].astype(int)

        df, user_mapping = convert_unique_idx(df, 'user')
        df, item_mapping = convert_unique_idx(df, 'item')
        
     

        df['count'] = df.groupby('user')['user'].transform('size')
        df = df.sort_values('count', ascending=False)
        df = df[df['count'] > 20]
        df['count'] = df.groupby('item')['item'].transform('size')
        df = df.sort_values('count', ascending=False)
        df = df[df['count'] >= 1]
        df = df[df['count'] <= 5]

        print('number of unique users = ', len(df['user'].unique()))
        print('number of unique items = ',len(df['item'].unique()))
        print('number of data points = ', len(df))
        

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

def write_data_to_file(idx1, idx2):
    df, item_mapping = LibraryThing('', idx2, idx1, idx2).load()
    df = df.reset_index()
    #df['rate'] = [int(i) for i in np.ceil(df['rate'])]
    print(df.head())
    print('writing')
    with open("ratings_lt.dat", "a") as f:
        for i in range(len(df)):
            f.write(str(df['user'][i])+"::"+str(df['item'][i])+"::"+str(df['rate'][i])+"::"+str(df['unixtime'][i])+"\n") #+"::"+str(df['nhelpful'][i])+"::"+str(df['commentlength'][i])+"::"+str(df['key_item'][i])+"::\n")


write_data_to_file(1, 1000000)
#write_data_to_file(1000000, 2000000)


def test_train_split(file):
    import random 

    input = open(file, 'r') 

    for line in input: 
        with open("train_lt.dat", 'a') as f:
            temp = line.split('::')
            f.write(temp[0] + '::' + temp[1] + '::' +temp[2] + '::' +temp[3] + '\n') 


    
#test_train_split('ratings_lt_correct.dat')
