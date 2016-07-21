from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.spaces import discrete, product, box

import re

BIG = 1e6
class Box2DEnvUB(Box2DEnv, Serializable):
    @autoargs.arg("frame_skip", type=int,
                  help="Number of frames to skip")
    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                  'problem non-Markovian!)')
    
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, filename=None, *args, **kwargs):
        super(Box2DEnvUB, self).__init__(*args, **kwargs)
        
        f = open("LaMnO3 reflections.txt", 'r')
        self.hkl_actions = []; count = 0
        self.obs = []
        for line in f: 
            count += 1
            intermed = line.split()
            if not re.search('^[^A-z]+$', intermed):
                self.hkl_actions.append(intermed[0:4]) #h, k, l, two_theta
                self.obs.append(intermed[11:13]) #Intensity and structure factor
        
        print self.hkl_actions
        self.hkl_space = discrete.Discrete(count)
        f.close()   
        
    @overrides
    def action_space(self):
        lbnd = np.array([-90, 0]); ubnd = np.array([90, 360])
        return box.Box(lbnd, ubnd)
    
    def discrete_action_space(self):
        return self.hkl_space
    
    @overrides
    def forward_dynamics(self, action):
        raise NotImplementedError
    
    @overrides
    def compute_reward(self, action, observation):
        """
        The implementation of this method should have two parts, structured
        like the following:

        <perform calculations before stepping the world>
        yield
        reward = <perform calculations after stepping the world>
        yield reward
        """
        raise NotImplementedError
    
    @overrides
    def step(self, action, observation):
        """   This is identical to the
        original step method, except it takes one more
        parameter (needed by compute_reward).    """
        
        reward_computer = self.compute_reward(action, observation)
        #forward the state
        action = self._inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.forward_dynamics(action)
        # notifies that we have stepped the world
        reward_computer.next()
        # actually get the reward
        reward = reward_computer.next()
        self._invalidate_state_caches()
        done = self.is_current_done()
        next_obs = self.get_current_obs(action)
        return Step(observation=next_obs, reward=reward, done=done)
    
    @overrides
    def get_raw_obs(self, action):
        """   Unlike in traditional physics problems, the
        observations are not positions and velocities
        Here, our observations are read directly from the machine
        """
        assert action[0] == 0
        return 100.0
    
    @overrides
    def get_current_obs(self, action):
        """ Identical to box2d_env's method, but with different paramters """
        raw_obs = self.get_raw_obs(action)
        noisy_obs = self._inject_obs_noise(raw_obs)
        if self.position_only:
            return self._filter_position(noisy_obs)
        return noisy_obs        
    
    @overrides
    def _get_position_ids(self):
        pass
