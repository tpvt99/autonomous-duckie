from train2 import *
# from train2_mid_actions import *
# from train2_few_actions import *

def test(model_class, MAP_NAME, MODEL_PATH=MODEL_PATH, seeds_dict=seeds_dict):
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(640, 480))
    display.start()
    torch.cuda.empty_cache()
    MODEL_PATH = os.path.join(MODEL_PATH, MAP_NAME)
    # MODEL_PATH = os.path.join(MODEL_PATH, '{}_27Mar'.format(MAP_NAME))
    # MODEL_PATH = os.path.join(MODEL_PATH, '{}_more_actions'.format(MAP_NAME))
    SEED = seeds_dict[MAP_NAME][0]  # just try the first seed

    # Initialize model and target network
    # model = load_model(os.path.join(MODEL_PATH, "run_{}-first-try.pt".format(50000)))
    model = load_model(os.path.join(MODEL_PATH, "run_{}.pt".format(2000)))
    model.eval()
    print("Model restored.")
    print(model)

    env = launch_env(MAP_NAME, SEED)
    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to rescale images
    env = ModifiedRewardWrapper(env)
    print("Initialized wrappers")

    # Set seeds
    seed(SEED)
    state_dim = env.observation_space.shape
    print("State size:", state_dim)
    # action_dim = env.action_space.shape[0]

    # Discretize action space. Or use DiscreteWrapper.
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    step_size = 0.2
    discrete_v = np.arange(0, max_action + step_size / 2, step_size)  # velocity. [0, 1]
    # discrete_v = [k for k in discrete_v if k != 0]
    step_size = 0.2
    discrete_w = np.arange(
        min_action, max_action + step_size / 2, step_size
    )  # steer angle [-1, 1]
    discrete_v = np.round(discrete_v, 1)
    discrete_w = np.round(discrete_w, 1)
    for w in range(len(discrete_w)):
        if discrete_w[w] == 0:
            discrete_w[w] = 0
    action_list = list(product(discrete_v, discrete_w))
    action_dim = len(action_list)
    print("Action space:", action_dim)


    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    epsilon = 0
    state = env.reset()
    outputs = []
    env.render()
    t = 0
    while True:
        print(t)
        add = np.concatenate([env.cur_pos, np.array([env.cur_angle])])
        
        action_idx = model.act(
            state=state,
            epsilon=epsilon,
            additional_data=add,
        )
        # print(action_idx)
        action = action_list[action_idx]
        outputs.append(action)
        

        # Apply the action to the environment
        next_state, reward, done, info = env.step(
            action
        )  # action has to be [speed, steering]

        # if episode == 0 and t == 0:
        #   import matplotlib.pyplot as plt
        #   # the resized obs
        #   plt.imshow(next_state[0:3,:,:].transpose([1,2,0]))
        #   plt.title('Observation resized')
        #   plt.show()
        #   # road lines obs
        #   plt.imshow(next_state[3:,:,:].transpose([1,2,0]))
        #   plt.title('Observation road lines')
        #   plt.show()
        
        # Save transition to replay buffer
        state = next_state
        rewards.append(reward)
        t += 1
        if done:
            break
        
    avg_rewards = np.mean(rewards)
    print("{} run(s) avg rewards : {:.1f}".format(t, avg_rewards))
    # display.sendstop()
    return outputs


if __name__ == "__main__":
    outputs = test(DeepQNetwork, MAP_NAME="map1")
    SEED = seeds_dict["map1"][0]
    # np.savetxt('map1_more_actions_seed{}.txt'.format(SEED), outputs, delimiter=',')
    # np.savetxt('map1_mid_actions_seed{}.txt'.format(SEED), outputs, delimiter=',')
    np.savetxt('map1_seed{}.txt'.format(SEED), outputs, delimiter=',')
    # np.savetxt('map1_seed{}.txt'.format(SEED), outputs, delimiter=',')