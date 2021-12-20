import ctypes
import time
import sys
from PIL import ImageGrab
from chainerrl import agents, distribution
from chainerrl.agents.ddpg import DDPG, DDPGModel
import numpy as np
import cv2
import pynput
#import cupy

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainerrl

SENDINPUT = ctypes.windll.user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wvk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlagss", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_ulong),
                ("dy", ctypes.c_ulong),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

#DDPGはActor-Criticを使うのでポリシーネットワークとQ関数で別のニューラルネットを用意する
#ポリシーネットワーク
class PolicyNetwork(chainer.Chain):
    def __init__(self):
        print("Initializing DQN...")
        print("Model Building")
        super(PolicyNetwork, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 8, 2, 3)
            self.conv2 = L.Convolution2D(16, 32, 5, 2, 2)
            self.conv3 = L.Convolution2D(32, 64, 5, 2, 2)
            self.lstm = L.LSTM(19200, 512)
            self.q = L.Linear(512, 18) #アクション数は18通り

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.lstm(h3)
        return distribution.ContinuousDeterministicDistribution(F.tanh(self.q(h4))) #連続値(18次元のベクトル)を返す

#Q関数
class QFunction(chainer.Chain):
    def __init__(self):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 8, 2, 3)
            self.conv2 = L.Convolution2D(16, 32, 5, 2, 2)
            self.conv3 = L.Convolution2D(32, 64, 5, 2, 2)
            self.lstm = L.LSTM(19200, 512)
            self.l1 = L.Linear(512, 100) #状態を100次元に変換
            self.l2 = L.Linear(100+18, 1) #100+18(18は状態次元数)

    def __call__(self, s, action):
        h1 = F.relu(self.conv1(s))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.lstm(h3)
        h5 = F.tanh(self.l1(h4))
        h6 = F.concat((h5, action), axis=1) #状態と行動を結合
        return self.l2(h6) #状態と行動からQ値を求める

def random_action():
    return np.random.randint(0, 17)

#Actual Functions

def presskey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SENDINPUT(1, ctypes.pointer(x), ctypes.sizeof(x))

def releasekey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SENDINPUT(1, ctypes.pointer(x), ctypes.sizeof(x))

def commandstart(action):
    if action == -1:
        presskey(0x01)#ESC
    elif action == 0:
        pass
    elif action == 1:
        presskey(0xcb)#LEFT
    elif action == 2:
        presskey(0xc8)#UP
    elif action == 3:
        presskey(0xcd)#RIGHT
    elif action == 4:
        presskey(0xd0)#DOWN
    elif action == 5:
        presskey(0xcb)#LEFT
        presskey(0xc8)#UP
    elif action == 6:
        presskey(0xc8)#UP
        presskey(0xcd)#RIGHT
    elif action == 7:
        presskey(0xcd)#RIGHT
        presskey(0xd0)#DOWN
    elif action == 8:
        presskey(0xd0)#DOWN
        presskey(0xcb)#LEFT
    elif action == 9:
        presskey(0x2a)#LSHIFT
    elif action == 10:
        presskey(0x2a)#LSHIFT
        presskey(0xcb)#LEFT
    elif action == 11:
        presskey(0x2a)#LSHIFT
        presskey(0xc8)#UP
    elif action == 12:
        presskey(0x2a)#LSHIFT
        presskey(0xcd)#RIGHT
    elif action == 13:
        presskey(0x2a)#LSHIFT
        presskey(0xd0)#DOWN
    elif action == 14:
        presskey(0x2a)#LSHIFT
        presskey(0xcb)#LEFT
        presskey(0xc8)#UP
    elif action == 15:
        presskey(0x2a)#LSHIFT
        presskey(0xc8)#UP
        presskey(0xcd)#RIGHT
    elif action == 16:
        presskey(0x2a)#LSHIFT
        presskey(0xcd)#RIGHT
        presskey(0xd0)#DOWN
    elif action == 17:
        presskey(0x2a)#LSHIFT
        presskey(0xd0)#DOWN
        presskey(0xcb)#LEFT

def commandend(action):
    if action == -1:
        releasekey(0x01)#ESC
    elif action == 0:
        #releasekey(0x2c)#Z
        pass
    elif action == 1:
        releasekey(0xcb)#LEFT
    elif action == 2:
        releasekey(0xc8)#UP
    elif action == 3:
        releasekey(0xcd)#RIGHT
    elif action == 4:
        releasekey(0xd0)#DOWN
    elif action == 5:
        releasekey(0xcb)#LEFT
        releasekey(0xc8)#UP
    elif action == 6:
        releasekey(0xc8)#UP
        releasekey(0xcd)#RIGHT
    elif action == 7:
        releasekey(0xcd)#RIGHT
        releasekey(0xd0)#DOWN
    elif action == 8:
        releasekey(0xd0)#DOWN
        releasekey(0xcb)#LEFT
    elif action == 9:
        #LSHIFT
        releasekey(0x2a)#LSHIFT
    elif action == 10:
        releasekey(0x2a)#LSHIFT
        releasekey(0xcb)#LEFT
    elif action == 11:
        releasekey(0x2a)#LSHIFT
        releasekey(0xc8)#UP
    elif action == 12:
        releasekey(0x2a)#LSHIFT
        releasekey(0xcd)#RIGHT
    elif action == 13:
        releasekey(0x2a)#LSHIFT
        releasekey(0xd0)#DOWN
    elif action == 14:
        releasekey(0x2a)#LSHIFT
        releasekey(0xcb)#LEFT
        releasekey(0xc8)#UP
    elif action == 15:
        releasekey(0x2a)#LSHIFT
        releasekey(0xc8)#UP
        releasekey(0xcd)#RIGHT
    elif action == 16:
        releasekey(0x2a)#LSHIFT
        releasekey(0xcd)#RIGHT
        releasekey(0xd0)#DOWN
    elif action == 17:
        releasekey(0x2a)#LSHIFT
        releasekey(0xd0)#DOWN
        releasekey(0xcb)#LEFT

def main():
    #ハイパーパラメーター
    GAMMA = 0.98
    NUM_EPISODE = 10000 #総試行回数
    state = ""

    #画像処理用
    DEATHCHECK_FILE = "Death.png"
    SCORE = "Score_4.png"
    TEN = "ttt.png"
    CHAPTER = "Chapter.png"
    IMG_D = cv2.imread(DEATHCHECK_FILE, cv2.IMREAD_COLOR) #被弾確認用
    IMG_S = cv2.imread(SCORE, cv2.IMREAD_COLOR) #P確認用
    IMG_T = cv2.imread(TEN, cv2.IMREAD_COLOR) #点確認用
    IMG_C = cv2.imread(CHAPTER, cv2.IMREAD_COLOR) #Chapter Finish確認用
    X = 284
    Y = 290
    W = 670
    H = 740 #撮影の座標指定

    #DQNのセットアップ
    q_func = QFunction()
    policy = PolicyNetwork()
    #q_func.to_gpu(0)
    model = DDPGModel(q_func=q_func, policy=policy)
    optimizer_p = optimizers.AdaDelta(rho=0.95, eps=1e-07)
    optimizer_q = optimizers.AdaDelta(rho=0.95, eps=1e-06)
    optimizer_p.setup(model["policy"])
    optimizer_q.setup(model["q_function"])
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.1, end_epsilon=0.0, decay_steps=NUM_EPISODE*100, random_action_func=random_action)
    replay_buffer = chainerrl.replay_buffer.EpisodicReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = chainerrl.agents.DDPG(model, optimizer_p, optimizer_q,  replay_buffer, GAMMA, explorer, gpu=0, replay_start_size=5000, minibatch_size=100, update_interval=50, target_update_interval=2000, phi=phi, episodic_update=True, episodic_update_len=50)
    agent.load("agent_TouhouAIDDPG_4700")

    try:
        command = 0
        reward = 0
        time_step = 0
        t = 0
        presskey(0x2c)#pressZ
        for episode in range(1, NUM_EPISODE + 1):
            time.sleep(1)
            commandstart(-1)
            time.sleep(1/60)
            commandend(-1)
            print("episode: {}".format(episode))
            done = False
            r = 0
            t = 0
            previous_p_check = 0
            previous_t_check = 0
            previous_time = 0
            time.sleep(1)
            time_hit = time.time()
            while not done:
                commandstart(command)
                time.sleep(1/60)
                img = ImageGrab.grab((X, Y, W, H))
                img = np.asarray(img, dtype="uint8")
                img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                match_result_d = cv2.matchTemplate(img_1, IMG_D, cv2.TM_CCOEFF_NORMED)
                match_result_p = cv2.matchTemplate(img_1, IMG_S, cv2.TM_CCOEFF_NORMED)
                match_result_t = cv2.matchTemplate(img_1, IMG_T, cv2.TM_CCOEFF_NORMED)
                match_result_c = cv2.matchTemplate(img_1, IMG_C, cv2.TM_CCOEFF_NORMED)
                death_check = np.column_stack(np.where(match_result_d >= 0.75))
                p_check = np.column_stack(np.where(match_result_p >= 0.5))
                t_check = np.column_stack(np.where(match_result_t >= 0.7))
                c_check = np.column_stack(np.where(match_result_c >= 0.8))
                if len(death_check) >= 1:
                    done = True
                    print("被弾")
                    reward = -100
                    print("reward: {}".format(reward))
                else:
                    img = cv2.resize(img, dsize=(120, 160))
                    #img = np.asarray(img.transpose(2, 0, 1), dtype=np.float32)
                    state = img.reshape(1, 160, 120)
                    new_command = agent.act_and_train(state, reward)
                    if(type(new_command) != int):
                        new_command = np.argmax(new_command)
                    print("new command: {}".format(new_command))
                    reward = 0
                    if time.time() - time_hit > 3:
                        if len(p_check) >= 1:
                            print("The number of P: {}".format(len(p_check)))
                            reward += max(len(p_check)-previous_p_check, 0)
                            previous_p_check = len(p_check)
                        if len(t_check) >= 1:
                            print("The number of Ten: {}".format(len(t_check)))
                            reward += max(len(t_check)-previous_t_check, 0)
                            previous_t_check = len(t_check)
                        if len(c_check) >= 1:
                            print("Chapter finished")
                            cool_time = time.time()
                            if cool_time - previous_time > 5:
                                reward += 100
                                previous_time = cool_time
                        print("reward: {}".format(reward))
                        r += reward
                    commandend(command) #ここまでコマンド入力
                    command = new_command
                t += 1
                if t > 500: #被弾検出出来なかったとき用
                    done = True
                    time.sleep(30/60)
                    commandstart(-1)
                    time.sleep(1/60)
                    commandend(-1)
            time_step += t
            print("time step: {}, average time step: {}".format(t, time_step/episode))
            if episode % 10 == 0:
                print("episode: {}, r: {}, statistics: {}".format(episode, r, agent.get_statistics())) #10エピソード毎にエピソード数、報酬の総和とその他の統計を表示
            agent.stop_episode_and_train(state, reward, done) #エピソードを終了して学習させる
            if episode % 100 == 0: #100エピソード毎にエージェントモデルを保存
                print("model saved")
                #agent.save("agent_TouhouAIDDPG_1_" + str(episode))
                agent.save("agent_TouhouAIDDPG_4800")

    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()
