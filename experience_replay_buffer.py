import numpy as np
import random

class ExperienceReplayBuffer(object):
    def __init__(self, size, alpha, beta):
        self._storage = []
        self._weights = []
        self._maxsize = size
        self._next_idx = 0
        self.alpha = alpha
        self.beta = beta
        self._deltas = np.zeros(self._maxsize)
        self._probs = np.zeros(self._maxsize)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, delta):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        
        self._deltas[self._next_idx] = delta ** self.alpha
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, weights = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            weights.append(1 / (self.__len__() * self._probs[i]) ** self.beta)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
            np.array(weights)
        )

    def sample(self, batch_size):
        self._probs = self._deltas / self._deltas.sum()
        idxes = [
            np.random.choice(range(self._maxsize), p=self._probs)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)
