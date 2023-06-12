# import math
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_additional_inputs=0):
        super().__init__()
        self.input_shape = input_shape  # [Channel, Height, Width]
        self.num_actions = num_actions  # Number of actions
        self.num_additional_inputs = num_additional_inputs
        self.construct()

    def feature_size(self):
        x = autograd.Variable(
            torch.zeros(1, *self.input_shape)
        )  # Variable with batch size of 1
        x = self.features(x)
        return x.view(1, -1).size(1) + self.num_additional_inputs

    def construct(self):
        # Feature map
        self.features = nn.Sequential(
            nn.Conv2d(
                self.input_shape[0], 32, kernel_size=3
            ),  # Input Channels, Output Channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.network = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, state, additional_data=None):
        state = self.features(torch.tensor(state).float().to(device))
        # state = state.view(state.size(0), -1)
        state = state.reshape(state.shape[0], -1)
        if additional_data is not None:
            additional_data = torch.tensor(additional_data).float().to(device)
            state = torch.cat((state, additional_data), dim=1)
        q_value = self.network(state)
        return q_value

    def act(self, state, additional_data=None, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if additional_data is not None:
            if not isinstance(additional_data, torch.FloatTensor):
                additional_data = (
                    torch.from_numpy(additional_data).float().unsqueeze(0).to(device)
                )
        if random.uniform(0.0, 1.0) < epsilon:
            action = random.randint(0, self.num_actions - 1)
            # random_v = random.uniform(0, 1)
            # random_w = random.uniform(-1, 1)
            # action = [random_w, random_w]
            return action
        action = self.forward(state, additional_data)
        action = int(action.argmax(1)[0]) # if discrete
        return action

def save_model(model, model_path):
    data = (
        model.__class__.__name__,
        model.state_dict(),
        model.input_shape,
        model.num_actions,
        # model.num_additional_inputs,
    )
    # torch.save(data, model_path, _use_new_zipfile_serialization=False)
    torch.save(data, model_path)


def load_model(model_path):
    (
        model_class,
        model_state_dict,
        input_shape,
        num_actions,
        # num_additional_inputs,
    ) = torch.load(model_path, map_location=torch.device(device))
    num_additional_inputs = 4
    model = eval(model_class)(input_shape, num_actions, num_additional_inputs).to(
        device
    )
    # model = eval(model_class)(input_shape, num_actions).to(
    #     device
    # )
    model.load_state_dict(model_state_dict)
    return model


def compute_loss(
    model, target, gamma, states, actions, actions_idx, rewards, next_states, dones, adds,
):
    batch_size = states.shape[0]
    done_mask = dones.squeeze()
    loss = nn.SmoothL1Loss()

    model_q_value = model.forward(states, adds).gather(1, actions_idx.reshape([-1, 1]))
    target_q_value = rewards.squeeze()    

    item = (
        gamma
        * target.forward(next_states[~done_mask], adds[~done_mask])
        .max(1)[0]
        .detach()
    )
    target_q_value[~done_mask] += item
    return loss(model_q_value, target_q_value.unsqueeze(1))


def optimize(model, target, memory, optimizer, batch_size, gamma):
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, gamma, **batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss