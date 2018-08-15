""" gym wrapper for rendering adversarial frames. """
import numpy as np
import gym
from gym import Wrapper

class AdversarialWrapper(Wrapper):

    def __init__(self, wrapper):
        super(AdversarialWrapper, self).__init__(wrapper.env)
        self.env._get_image = self._getAdvImage
        self.setAdvDiff(None)
        
    def setAdvDiff(self, advDiff):
        self._adv_diff = advDiff

    def _getAdvImage(self):
        if self._adv_diff is None:
            img = self.env.ale.getScreenRGB2()
        else:
            img = self.env.ale.getScreenRGB2() + self._adv_diff

        return img

    def render(self, mode='human', **kwargs):
        #advDiff = kwargs.get('advDiff')
        #self.setAdvDiff(advDiff)

        return self.env.render(mode)


#env = gym.make('PongNoFrameskip-v4')
#advEnv = AdversarialWrapper(env)
#advEnv.reset()
#a = np.random.randint(0, 20, advEnv.observation_space.shape, dtype=np.uint8)
