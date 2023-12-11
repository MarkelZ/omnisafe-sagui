from omnisafe.envs.sagui_envs import register_sagui_envs
from pynput.keyboard import Listener, KeyCode
import safety_gymnasium
from time import sleep


class InputAction:
    def __init__(self):
        self.keys = {'w': False, 'a': False, 's': False, 'd': False}
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if isinstance(key, KeyCode):
            k_char = key.char
            if k_char in self.keys:
                self.keys[k_char] = True

    def on_release(self, key):
        if isinstance(key, KeyCode):
            k_char = key.char
            if k_char in self.keys:
                self.keys[k_char] = False

    def get_act_from_input(self):
        act = [0, 0]
        if self.keys['w']:
            act[0] += 1
        if self.keys['s']:
            act[0] -= 1
        if self.keys['a']:
            act[1] += 1
        if self.keys['d']:
            act[1] -= 1
        return act


register_sagui_envs()

env = safety_gymnasium.make('SafetyPointGuide0-v0', render_mode='human')
obs, info = env.reset()

input_act = InputAction()

terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
while True:
    sleep(0.016)
    if terminated or truncated:
        print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
        ep_ret, ep_cost = 0, 0
        obs, info = env.reset()

    act = input_act.get_act_from_input()
    obs, reward, cost, terminated, truncated, info = env.step(act)

    ep_ret += reward
    ep_cost += cost
