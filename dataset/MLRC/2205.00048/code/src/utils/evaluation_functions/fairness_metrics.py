import torch
import time

def II_F(E_system, E_target, E_collect, batch_indicator):
    print('initialized')
    start_temp = time.time()
    # the batch_indicator is a matrix, where 0: hold-out; 1: should consider
    E_system = E_system - E_collect
    E_target = E_target - E_collect
    metric = (E_system - E_target).pow(2).sum() / batch_indicator.sum()
    dis = (E_system).pow(2).sum() / batch_indicator.sum()
    rel = 2 * (E_system * E_target).sum() / batch_indicator.sum()
    stop_temp = time.time()
    print('Time IIF: ', stop_temp - start_temp)
    return [metric, dis, rel]


def GI_F(E_system, E_target, E_collect, user_label, batch_indicator):
    start_temp = time.time()
    E_system = (E_system - E_collect).double()
    E_target = (E_target - E_collect).double()
    num_userG = user_label.shape[0]
    num_item = E_system.shape[1]    
    user_label = user_label.double()
    batch_indicator = batch_indicator.double()
    metric, dis, rel = 0, 0, 0

    for i in range(num_userG):
        #print('-------')
        #print(user_label.shape)
        #print(user_label[i].shape)
        #print(user_label[i].view(-1, 1).shape)
        #print(E_system.shape)
        
        diff = (E_system * user_label[i].view(-1, 1) - E_target * user_label[i].view(-1, 1)).sum(0, keepdim=True)
        #print(diff.shape)
        #print('.......')
        dis_tmp = (E_system * user_label[i].view(-1, 1)).sum(0, keepdim=True)
        rel_tmp = (E_system * user_label[i].view(-1, 1)).sum(0, keepdim=True) * (
                E_target * user_label[i].view(-1, 1)).sum(
            0, keepdim=True)
        num = (batch_indicator * user_label[i].view(-1, 1)).sum(0, keepdim=True)
        
        num[num == 0] = 1

        metric += (diff / num).pow(2).sum()
        dis += (dis_tmp / num).pow(2).sum()
        rel += (rel_tmp / num / num).sum()
        
    metric = metric / num_userG / num_item
    dis = dis / num_userG / num_item
    rel = 2 * rel / num_userG / num_item
    stop_temp = time.time()

    print('Time GIF: ', stop_temp - start_temp)
    return [metric, dis, rel]


def IG_F(E_system, E_target, E_collect, item_label, batch_indicator):
    start_temp = time.time()
    E_system = (E_system - E_collect).double()
    E_target = (E_target - E_collect).double()
    num_user = E_system.shape[0]
    num_itemG = item_label.shape[0]
    metric, dis, rel = 0, 0, 0

    

    for i in range(num_itemG):
        diff = (E_system * item_label[i] - E_target * item_label[i]).sum(1, keepdim=True)
        dis_tmp = (E_system * item_label[i]).sum(1, keepdim=True)
        rel_tmp = (E_system * item_label[i]).sum(1, keepdim=True) * (E_target * item_label[i]).sum(1, keepdim=True)
        num = (batch_indicator * item_label[i]).sum(1, keepdim=True)
        num[num == 0] = 1

        metric += (diff / num).pow(2).sum()
        dis += (dis_tmp / num).pow(2).sum()
        rel += (rel_tmp / num / num).sum()

    metric = metric / num_user / num_itemG
    dis = dis / num_user / num_itemG
    rel = 2 * rel / num_user / num_itemG
    stop_temp = time.time()
    print('Time IGF: ', stop_temp - start_temp)
    return [metric, dis, rel]


def GG_F(E_system_raw, E_target_raw, E_collect, user_label, item_label, batch_indicator):
    start_temp = time.time()
    E_system = E_system_raw - E_collect
    E_target = E_target_raw - E_collect
    num_userG = user_label.shape[0]
    num_itemG = item_label.shape[0]
    metric, dis, rel = 0, 0, 0

    # GG_diff_matrix = torch.zeros(num_userG, num_itemG)
    GG_target_matrix = torch.zeros(num_userG, num_itemG)
    GG_system_matrix = torch.zeros(num_userG, num_itemG)
    GG_coll_matrix = torch.zeros(num_userG, num_itemG)
    

    for i in range(num_userG):
        for j in range(num_itemG):

            diff = ((E_system * user_label[i].view(-1, 1) - E_target * user_label[i].view(-1, 1)) * \
                    item_label[j]).sum()
            dis_tmp = ((E_system * user_label[i].view(-1, 1)) * item_label[j]).sum()
            rel_tmp = ((E_system * user_label[i].view(-1, 1)) * item_label[j]).sum() * (
                    (E_target * user_label[i].view(-1, 1)) * item_label[j]).sum()

            num = ((batch_indicator * user_label[i].view(-1, 1)) * item_label[j]).sum()
            num[num == 0] = 1

            metric += (diff / num).pow(2).sum()
            dis += (dis_tmp / num).pow(2).sum()
            rel += (rel_tmp / num / num).sum()
            # GG_diff_matrix[i][j] = diff.item()
            GG_target_matrix[i][j] = (E_target_raw * user_label[i].view(-1, 1) * item_label[j]).sum() / num
            GG_system_matrix[i][j] = (E_system_raw * user_label[i].view(-1, 1) * item_label[j]).sum() / num
            GG_coll_matrix[i][j] = (E_collect * user_label[i].view(-1, 1) * item_label[j]).sum() / num

    metric = metric / num_userG / num_itemG
    dis = dis / num_userG / num_itemG
    rel = 2 * rel / num_userG / num_itemG
    stop_temp = time.time()
    print('Time GGF: ', stop_temp - start_temp)
    return [metric, dis, rel, GG_target_matrix, GG_system_matrix, GG_coll_matrix]


def AI_F(E_system, E_target, E_collect, batch_indicator):
    start_temp = time.time()
    E_system = E_system - E_collect
    E_target = E_target - E_collect
    num_user = E_system.shape[0]
    num_item = E_system.shape[1]

    metric = ((E_system * batch_indicator).sum(0) - (E_target * batch_indicator).sum(0))
    dis = (E_system * batch_indicator).sum(0)
    rel = 2 * (E_system * batch_indicator).sum(0) * (E_target * batch_indicator).sum(0)
    num = batch_indicator.sum(0)
    num[num == 0] = 1

    metric = (metric / num).pow(2).sum() / num_item
    dis = (dis / num).pow(2).sum() / num_item
    rel = (rel / num / num).sum() / num_item
    stop_temp = time.time()
    print('Time AIF: ', stop_temp - start_temp)

    return [metric, dis, rel]


def AG_F(E_system, E_target, E_collect, item_label, batch_indicator):
    start_temp = time.time()
    E_system = E_system - E_collect
    E_target = E_target - E_collect
    num_user = E_system.shape[0]
    num_itemG = item_label.shape[0]
    metric, dis, rel = 0, 0, 0

    for i in range(num_itemG):
        diff = (E_system * batch_indicator * item_label[i]).sum() - (E_target * batch_indicator * item_label[i]).sum()
        dis_tmp = (E_system * batch_indicator * item_label[i]).sum()
        rel_tmp = 2 * (E_system * batch_indicator * item_label[i]).sum() * (
                E_target * batch_indicator * item_label[i]).sum()
        num = (batch_indicator * item_label[i]).sum()
        num[num == 0] = 1

        metric += (diff / num).pow(2)
        dis += (dis_tmp / num).pow(2)
        rel += (rel_tmp / num / num).sum()
       
    metric = metric / num_itemG
    dis = dis / num_itemG
    rel = rel / num_itemG
    stop_temp = time.time()
    print('Time AGF: ', stop_temp - start_temp)
    return [metric, dis, rel]

#reproduces figure 1 example
if __name__ == '__main__':
    """ Run Toy Example 
    """

    E_system1 = torch.tensor([[0.5, 0.0, 0.5, 0.0],
                              [0.0, 0.5, 0.0, 0.5],
                              [0.5, 0.0, 0.5, 0.0],
                              [0.0, 0.5, 0.0, 0.5]]).float()
    E_system2 = torch.tensor([[0.5, 0.5, 0, 0],
                              [0, 0, 0.5, 0.5],
                              [0.5, 0.5, 0, 0],
                              [0, 0, 0.5, 0.5]]).float()
    E_system3 = torch.tensor([[0.5, 0, 0.5, 0],
                              [0.5, 0, 0.5, 0],
                              [0, 0.5, 0, 0.5],
                              [0, 0.5, 0, 0.5]]).float()
    E_system4 = torch.tensor([[0.5, 0, 0.5, 0],
                              [0.5, 0, 0.5, 0],
                              [0.5, 0, 0.5, 0],
                              [0.5, 0, 0.5, 0]]).float()
    E_system5 = torch.tensor([[0.5, 0.5, 0, 0],
                              [0.5, 0.5, 0, 0],
                              [0, 0, 0.5, 0.5],
                              [0, 0, 0.5, 0.5]]).float()
    E_system6 = torch.tensor([[0.5, 0.5, 0, 0],
                              [0.5, 0.5, 0, 0],
                              [0.5, 0.5, 0, 0],
                              [0.5, 0.5, 0, 0]]).float()
    E_system7 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                              [0.2, 0.3, 0.3, 0.2],
                              [0.1, 0.4, 0.2, 0.3],
                              [0.1, 0.3, 0.2, 0.4]]).float()
    E_target = torch.tensor([[0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25]]).float()
    indicator = torch.tensor([[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]).float()

    item_label = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]).long()
    user_label = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]).long()

    E_system_list = [E_system1, E_system2, E_system3, E_system4, E_system5, E_system6, E_system7]

    E_collect = torch.tensor([[0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25]]).float()

    for i in range(6):
        E_system = E_system_list[i] * indicator
        E_target = E_target * indicator

        print("situation {}:".format(i + 1))

        print("II-F:", II_F(E_system, E_target, E_collect, indicator))
        print("IG-F:", IG_F(E_system, E_target, E_collect, item_label, indicator))
        print("GI-F:", GI_F(E_system, E_target, E_collect, user_label, indicator))
        print("GG-F:", GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[:3])
        print("AI-F:", AI_F(E_system, E_target, E_collect, indicator))
        print("AG-F:", AG_F(E_system, E_target, E_collect, item_label, indicator))
