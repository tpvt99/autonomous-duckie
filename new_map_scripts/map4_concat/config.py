import torch

seeds_dict = {
    "map1": [2, 3, 5, 9, 12],
    "map2": [1, 2, 3, 5, 7, 8, 13, 16],
    "map3": [1, 2, 4, 8, 9, 10, 15, 21],
    "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
    "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}


debug = False
if not debug:
    # to tune this
    max_episodes = 200000
    MAX_STEPS = 2000  # set as a large number for training
    t_max = 1500
    min_buffer = 99000
    epsilon_decay = 32000
    buffer_limit = 150000
    bias = 10
    target_update = 500  # episode(s) # 100

    # no need to edit
    SCALING_FACTOR = 0.125
    learning_rate = 0.0001
    gamma = 0.98
    batch_size = 32
    print_interval = 20
    save_interval = 500
    train_steps = 10

    restore = False
    to_save = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert buffer_limit >= min_buffer

else:
     # to tune this
    max_episodes = 200000
    MAX_STEPS = 2000  # set as a large number for training
    t_max = 1500
    min_buffer = 5
    epsilon_decay = 32000
    buffer_limit = 100000
    bias = 10
    target_update = 1  # episode(s) # 100

    # no need to edit
    SCALING_FACTOR = 0.125
    learning_rate = 0.0001
    gamma = 0.98
    batch_size = 32
    print_interval = 20
    save_interval = 500
    train_steps = 10

    restore = True
    to_save = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert buffer_limit >= min_buffer

