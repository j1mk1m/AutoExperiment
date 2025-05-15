from collections import Counter
import torch

def pop_index(df):
    # count number of reviews per movie
    count = Counter(df['item'])
    common = count.most_common() #returns list of tuples of (element, count) sorted by counts(mostly commonly viewed rated movie)
    item_size = len(set(df['item'])) # = len(common) = 3706

    index_pop = [[], [], [], [], []]
    
    for i in range(item_size):
        #compares the number of occurences of each movie with an occurence in common idk why
        if count[i] > common[int(0.2 * len(common))][1]:
            index_pop[0].append(i)
        elif count[i] > common[int(0.4 * len(common))][1]:
            index_pop[1].append(i)
        elif count[i] > common[int(0.6 * len(common))][1]:
            index_pop[2].append(i)
        elif count[i] > common[int(0.8 * len(common))][1]:
            index_pop[3].append(i)
        else:
            index_pop[4].append(i)
  
    for i in range(len(index_pop)):
        index_pop[i] = torch.tensor(index_pop[i])

    pop_size = 5
    pop_mask = torch.zeros(pop_size, item_size) #[5, 3706] 
    for i in range(item_size):
        for k in range(pop_size):
            if i in index_pop[k]:
                pop_mask[k][i] = 1

    return index_pop, pop_mask
