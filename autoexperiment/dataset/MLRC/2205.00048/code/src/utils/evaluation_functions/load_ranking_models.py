import pandas as pd 

def load_deterministic_ranker(args, item_mapping):
    """ Load pretrained deterministic ranker 
    """
    
    if args.data == 'ml-1m':
        save_df = pd.read_csv('src/models/ml/run-{}-ml-1M-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    elif args.data == 'lt':
        save_df = pd.read_csv('src/models/runs-libraryThing/run-{}-libraryThing-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    

    save_df = save_df.rename(columns={0: "user", 2: "item", 4: "score"})
    if args.data == 'ml-1m':
        save_df.user = save_df.user - 1

    save_df['item'] = save_df['item'].map(item_mapping)
    save_df = save_df.dropna().reset_index(drop = True)
    save_df.drop(save_df.tail(len(save_df)%100).index, inplace = True)
    
    
    save_df = save_df.sort_values(["user", "score"], ascending=[True, False])
    save_df = save_df.reset_index().drop(['index'], axis=1)
    return save_df
