import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QTrainer:
    def __init__(self, model, lr, gamma, device, num_channels, attack_eps):
        super(QTrainer, self).__init__()

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device

        # Move model to device
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # -----------------------------
        # ImageNet normalization params
        # -----------------------------
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)

        if num_channels == 1:
            self.imagenet_mean = np.mean(self.imagenet_mean)
            self.imagenet_std = np.mean(self.imagenet_std)

        # ✅ Device-safe tensors
        self.mu = torch.tensor(self.imagenet_mean, device=self.device).view(num_channels, 1, 1)
        self.std = torch.tensor(self.imagenet_std, device=self.device).view(num_channels, 1, 1)

        self.upper_limit = (1 - self.mu) / self.std
        self.lower_limit = (0 - self.mu) / self.std

        # Attack parameters
        self.attack_eps = attack_eps / self.std
        self.attack_step = (1.25 * attack_eps) / self.std
        self.ctr = 0

    # =========================================================
    # Training step
    # =========================================================
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        # Handle short memory case
        if state.shape[0] == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Predict next state values
        self.model.eval()
        with torch.no_grad():
            next_pred = self.model(next_state)

        # Current prediction
        self.model.train()
        pred = self.model(state)

        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx] and idx != len(done) - 1:
                Q_new = reward[idx] + self.gamma * torch.max(next_pred[idx])

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # =========================================================
    # Utility: clamp tensor
    # =========================================================
    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    # =========================================================
    # Create adversarial state (FGSM)
    # =========================================================
    def create_adv_state(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = torch.unsqueeze(state, 0)

        # Predict action
        self.model.eval()
        with torch.no_grad():
            pred = self.model(state)

        final_move = [0, 0, 0]
        move = torch.argmax(pred).item()
        final_move[move] = 1
        final_move = torch.tensor(final_move, dtype=torch.float32, device=self.device)

        # Create perturbation
        delta = torch.zeros_like(state, device=self.device)

        for j in range(len(self.attack_eps)):
            eps_val = self.attack_eps[j][0][0].item()
            delta[:, j, :, :].uniform_(-eps_val, eps_val)

        delta.data = self.clamp(delta, self.lower_limit - state, self.upper_limit - state)
        delta.requires_grad = True

        # Forward pass with perturbation
        self.model.train()
        output = self.model((state + delta[:state.size(0)]).float())

        self.optimizer.zero_grad()
        loss = self.criterion(output, final_move)
        loss.backward()

        grad = delta.grad.detach()

        delta.data = self.clamp(
            delta + self.attack_step * torch.sign(grad),
            -self.attack_eps,
            self.attack_eps,
        )

        delta.data[:state.size(0)] = self.clamp(
            delta[:state.size(0)],
            self.lower_limit - state,
            self.upper_limit - state,
        )

        self.ctr += 1
        return delta[:state.size(0)]
