import random
import torch
def random_split(N, ratio=(0.6, 0.2, 0.2), random_seed: int = 0):
    random.seed(random_seed)
    data_ids = [i for i in range(N)]
    train_mask, valid_mask, test_mask = torch.zeros(N), torch.zeros(N), torch.zeros(N)
    train_size = int(N * ratio[0])
    valid_size = int(N * ratio[1])
    random.shuffle(data_ids)

    train_ids = data_ids[:train_size]
    valid_ids = data_ids[train_size:(train_size+valid_size)]
    test_ids = data_ids[(train_size+valid_size):]
    train_mask[train_ids] = 1
    valid_mask[valid_ids] = 1
    test_mask[test_ids] = 1
    assert train_mask.sum() == train_size
    return train_mask, valid_mask, test_mask

def multi_round_split(N, round = 10):
    split_res = []
    for i in range(round):
        train, valid, test = random_split(N=N, random_seed=i)
        split_res.append((train, valid, test))
    return split_res


if __name__ == '__main__':
    N = 10
    split_res = multi_round_split(N=N, round=10)
    for train, val, test in split_res:
        print(train)
        print(val)
        print(test)
        print('*'*75)