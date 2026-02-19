import os
import datetime
import argparse
import random
import numpy as np
import cv2
import torch
from collections import deque
from game.doodlejump import DoodleJump
from model.networks import Deep_QNet, Deep_RQNet, DQ_Resnet18, DQ_Mobilenet, DQ_Mnasnet
from model.deepQTrainer import QTrainer
from helper import write_model_params
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, args):
        self.n_games = 0
        self.ctr = 1
        seed = args.seed
        self.exploration = args.exploration

        # -------------------------
        # ✅ Safe seeding
        # -------------------------
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.store_frames = args.store_frames
        self.image_h = args.height
        self.image_w = args.width
        self.image_c = args.channels
        self.memory = deque(maxlen=args.max_memory)

        # ✅ Device-safe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.steps = 0
        self.exploration_type = args.explore
        self.decay_factor = args.decay_factor
        self.epsilon = args.epsilon
        self.eulers_constant = 2.71828

        if args.explore == "epsilon_g_decay_exp":
            self.epsilon = 1

        # -------------------------
        # ✅ Model selection
        # -------------------------
        if args.model == "dqn":
            self.model = Deep_QNet()
        elif args.model == "drqn":
            self.model = Deep_RQNet()
        elif args.model == 'resnet':
            self.model = DQ_Resnet18()
        elif args.model == 'mobilenet':
            self.model = DQ_Mobilenet()
        elif args.model == 'mnasnet':
            self.model = DQ_Mnasnet()

        # Move model to device
        self.model.to(self.device)

        # -------------------------
        # ✅ Load pretrained weights safely
        # -------------------------
        if args.model_path:
            self.model.load_state_dict(
                torch.load(args.model_path, map_location=self.device)
            )
            print(f"Loaded model from {args.model_path} on {self.device}")

        # Trainer
        self.trainer = QTrainer(
            model=self.model,
            lr=self.lr,
            gamma=self.gamma,
            device=self.device,
            num_channels=self.image_c,
            attack_eps=args.attack_eps
        )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    def preprocess(self, state):
        img = cv2.resize(state, (self.image_w, self.image_h))
        M = cv2.getRotationMatrix2D((self.image_w / 2, self.image_h / 2), 270, 1.0)
        img = cv2.warpAffine(img, M, (self.image_h, self.image_w))

        if self.store_frames:
            os.makedirs("./image_dump", exist_ok=True)
            cv2.imwrite(f"./image_dump/{self.ctr}.jpg", img)
            self.ctr += 1

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if self.image_c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = ((img / 255.0) - np.mean(imagenet_mean)) / np.mean(imagenet_std)
        else:
            img = ((img / 255.0) - imagenet_mean) / imagenet_std
            img = img.transpose((2, 0, 1))

        img = np.expand_dims(img, axis=0)
        return img

    def get_state(self, game):
        state = game.getCurrentFrame()
        return self.preprocess(state)

    # --------------------------------------------------
    # Memory
    # --------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.model.train()
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.model.train()
        return self.trainer.train_step(state, action, reward, next_state, done)

    # --------------------------------------------------
    # Exploration
    # --------------------------------------------------
    def should_explore(self, test_mode):
        self.steps += 1
        if test_mode:
            return False

        r = random.random()

        if self.exploration_type == "epsilon_g_decay_exp":
            self.epsilon *= (1.0 - self.decay_factor) ** self.steps
        elif self.exploration_type == "epsilon_g_decay_exp_cur":
            self.epsilon = self.decay_factor * pow(self.eulers_constant, -self.steps)

        return r > self.epsilon

    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------
    def get_action(self, state, test_mode=False):
        final_move = [0, 0, 0]

        if self.should_explore(test_mode):
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


# =========================================================
# TRAIN
# =========================================================
def train(game, args, writer):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    agent = Agent(args)
    dummy_input = torch.rand(1, args.channels, args.height, args.width).to(agent.device)
    writer.add_graph(agent.model, dummy_input)

    print("Now training...")

    record = 0
    total_score = 0

    while agent.n_games != args.max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.playStep(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, [done])
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.gameReboot()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("model_best.pth")

            print(f'Game {agent.n_games} Score {score} Record {record}')

            total_score += score
            mean_score = total_score / agent.n_games
            writer.add_scalar('Score/Mean', mean_score, agent.n_games)


# =========================================================
# TEST
# =========================================================
def test(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    agent = Agent(args)
    print("Now testing...")

    record = 0
    cum_score = 0

    while agent.n_games < args.max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, test_mode=True)
        reward, done, score = game.playStep(final_move)

        if done:
            agent.n_games += 1
            cum_score += score
            game.gameReboot()

            if score > record:
                record = score

            print(f'Game {agent.n_games} Score {score} Record {record} Mean {cum_score/agent.n_games}')


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-d", "--difficulty", default="EASY")
    parser.add_argument("-m", "--model", default="dqn")
    parser.add_argument("-p", "--model_path", type=str)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-g", "--gamma", type=float, default=0.9)
    parser.add_argument("--max_memory", type=int, default=10000)
    parser.add_argument("--store_frames", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--reward_type", type=int, default=1)
    parser.add_argument("--exploration", type=int, default=40)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_games", type=int, default=100)
    parser.add_argument("--explore", default="epsilon_g")
    parser.add_argument("--decay_factor", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--attack_eps", type=float, default=0.3)
    args = parser.parse_args()

    game = DoodleJump(difficulty=args.difficulty, server=args.server, reward_type=args.reward_type)

    if args.human:
        game.run()
    elif args.test:
        test(game, args)
    else:
        writer = SummaryWriter(log_dir="runs")
        train(game, args, writer)
