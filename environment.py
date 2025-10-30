import numpy as np
import cv2

class AtariEnv(object):
    """
    A wrapper of the original Gym environment class.
    Applies preprocessing and frame stacking.
    """
    def __init__(self, env, frame_seq, frame_skip, screen_size=(42, 42), render=False):
        self.env = env
        self.screen_size = screen_size
        self.frame_skip = frame_skip
        self.frame_seq = frame_seq
        self.render = render
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.count_ = 0

    @property
    def state_shape(self):
        return [self.frame_seq, self.screen_size[0], self.screen_size[1]]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def process_image(self, frame):
        # Crop, resize twice, grayscale, normalize
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, self.screen_size)
        frame = frame.mean(2)  # grayscale
        frame = frame.astype(np.float32) * (1.0 / 255.0)
        frame = np.reshape(frame, [1, self.screen_size[0], self.screen_size[1]])
        return frame

    def reset_env(self):
        obs = self.env.reset()
        frame = self.process_image(obs)
        self.state = np.repeat(frame, self.frame_seq, axis=0)
        return self.state

    def forward_action(self, action):
        reward = 0
        done = False
        obs = None

        for _ in range(self.frame_skip):
            if self.render:
                self.env.render()

            # Handle both Gym v0.25 and newer versions
            result = self.env.step(action)
            if len(result) == 5:
                obs, r, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                obs, r, done, _ = result

            reward += r
            self.count_ += 1

            if done:
                break

        obs = self.process_image(obs)
        self.state = np.append(self.state[1:, :, :], obs, axis=0)
        reward = np.clip(reward, -1, 1)
        return self.state, reward
