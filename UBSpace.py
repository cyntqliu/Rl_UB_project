#UB matrix action space
from .base import Space
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
import re

from rllab.misc import autoargs
from rllab.misc.overrides import overrides

import numpy as np

class UBSpace(Space):
    
    @overrides
    def __init__(self,fname):
        self.fname = fname
        assert type(fname) is str, "Your file name is not a string"
        #print fname[-4:]
        if fname[-4:] == ".txt":
            f = open(fname, 'r')
            self.hkl_actions = []; count = 0
            self.obs = []
            for line in f: 
                count += 1
                line = line.strip()
                intermed = line.split()
                if re.search('^[^A-z]+$', line) and len(intermed) > 1:
                    self.hkl_actions.append([int(intermed[0]), int(intermed[1]), int(intermed[2])]) #h, k, l - thetas are calculated
                    self.obs.append(intermed[11:13]) #Intensity and structure factor
    
            self.hkl_actions = np.array(self.hkl_actions)
            self.obs = np.array(self.obs)
            self.hkl_discrete = Discrete(len(self.hkl_actions))
            self.last_discrete = 0 #index of the last discrete action
            f.close()
            
            #Now define entire space as if continuous
            lbnd = np.array([0, -90, 0, 0]); ubnd = np.array([-1, 90, 360, self.hkl_discrete.n])
            self.all_space = Box(lbnd, ubnd)
        
        else:
            raise Exception("The file is not of type .txt. If it is, please type the file extension in its name. ")
        
    def get_hkl_actions(self):
        return self.hkl_actions
    
    def get_discrete(self):
        return self.hkl_discrete
    
    def get_all_actions(self):
        return self.all_space
    
    def get_last_discrete(self):
        return self.last_discrete
    
    def get_obs(self):
        return self.obs
    
    @property
    def flat_dim(self):
        return 4
    
    @property
    def shape(self):
        temp = np.array([1.0, 2.0, 3.0, 4.0])
        return temp.shape
    
    @property
    def bounds(self):
        return self.all_space.bounds
    
    @overrides
    def contains(self, x):
        try:
            x = [i for i in x]
            assert type(x) is list, "Your action was iterable but it could not be converted into a list"
            
            if x[0] == 0: #discrete
                action = x[1:]
                for i in range(self.hkl_discrete.n):
                    if action[2] == i: return True
                return False
            elif x[0] == 1: #continuous
                action = x[1:]
                return action[0] >= -90 and action[0] <= 90 and action[1] >= 0 and action[1] <= 360
                
        except:
            raise Exception("Your action is invalid because it is not a list or an array")
        
    @overrides
    def __repr__(self):
        classInfo = "Possible hkl actions: " + str(self.hkl_actions) + " and then the continuous space " + str(self.cont_space)
        return classInfo
    
    @overrides
    def flatten(self, x):
        #copied from Box
        return np.asarray(x).flatten()
    
    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)
    
    @overrides
    def flatten_n(self, xs):
        #copied from Box
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0],) + self.shape)    
    
    @overrides
    def new_tensor_variables(self, name, extra_dims):
        pass
