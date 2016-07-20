from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

from rllab.spaces import base, discrete, product

BIG = 1e6
class Box2DEnvUB(Box2DEnv, Serializable):
    @autoargs.arg("frame_skip", type=int,
                  help="Number of frames to skip")
    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                  'problem non-Markovian!)')
    
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        super(Box2DEnvUB, self).__init__(*args, **kwargs)
        
    @property
    @overrides
    def _state(self):
        pass
        
    @overrides
    def action_space(self):
        pass
    
    @overrides
    def forward_dynamics(self, action):
        pass
    
    @overrides
    def step(self, action):
        pass
    
    @overrides
    def get_raw_obs(self):
        pass
    
    @overrides
    def _get_position_ids(self):
        pass
