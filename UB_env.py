import numpy as np
import math

from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env_ub import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
#import rllab.algos.ddpg_with_explore as dwe

import ubmatrix as ub
import lattice_calculator_procedural2 as lcp

import time
import sys

class UBEnv(Box3DEnv, Serializable):
    
    
    
    @autoargs.inherit(Box3DEnv.__init__)
    #Parent is Box2DEnv
    def __init__(self, *args, **kwargs):
        """    Constants:
        omega is always 0, and set a constant for background noise
        """
        
        self.Om = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.background = .01 #background noise
        
        """    Variables:
        These variables will be read along with the action:
        two_theta: detector's rotation about the z-axis -- assume elastic scattering, so omega is always 0
        theta: the angle at which our neutrons strike the plane
        
        These variables are the two dimensions of our problem
        chi: outer ring's rotation about the x-axis
        phi: rotation of the eulerian cradle, varies between z- and y-axis rotation depending on how much chi rotated
        
        """
        
        self.max_two_theta = 360
        self.max_chi = 90
        self.max_phi = 360
        super(UBEnv, self).__init__(self.model_path("UB.xml.mako"),
                                    *args, **kwargs)
        self.time = time.time() #Time cost
        
        #Two independent bodies
        #self.detector = find_body(self.world, "detector") #hkl determines two theta
        #self.base = find_body(self.world, "base") #omega
        self.ring = find_body(self.world, "ring") #chi
        self.eu_cradle = find_body(self.world, "eu_cradle") #phi
        
        self.ubs = []; self.U_mat = np.zeros(shape=(3,3))
        Serializable.__init__(self, *args, **kwargs)
        
    
    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        
        """      Variables:
        
        time: used to measure the time cost of the experiment
        ubs: a list of ubs, with a new ub calculated after every measurement
        wavelength: the wavelength used in experiment
        h, k, l: Miller indices. Take the place of x, y, z coordinates
        two_theta, omega, chi, phi: machine angles
        """
        
        self.time = time.time()
        ub_0, U_0, self.chi, self.phi = self.init_ub()
        self.ubs.append(ub_0)
        self.U_mat = np.array(U_0)
        
        good = False
        while good != True:
            try: 
                self.wavelength = float(input("What wavelength are you using for this experiment? "))
                good = True
            except:
                print "I'm sorry, please input a valid numerical value."
        
        """  Conversion Table
        
        Positions
        Only chi and phi are "independent," because the program has to figure them out using the UB matrix
        h, k, l represent an action
        two_theta and theta are read along with h, k, l

        Velocities are irrelevant
        """        
        self.ring.position = (0,self.ring.position[1])
        self.ring.linearVelocity = (self.ring.linearVelocity[0], self.ring.linearVelocity[1])
        self.eu_cradle.position = (self.eu_cradle[0],0)
        self.eu_cradle.linearVelocity = (self.eu_cradle.linearVelocity[0], self.eu_cradle.linearVelocity[1])     
        
        return self.get_current_obs(), ub_0
    
    @overrides
    def compute_reward(self, action, observation):
        yield
        timeCost = (time.time() - self.time)/1000
        if observation[0] >= self.background:
            accuracy = 1.0
        else: accuracy = 0.0
        
        exp_chi, exp_phi = self.calc_expected(action)
        loss = self.calc_loss(self, exp_chi, exp_phi)
        
        reward = accuracy - timeCost - loss/100.0
        yield reward
    
    @overrides
    def is_current_done(self,action):
        exp_chi, exp_phi = self.calc_expected(action)
        loss = self.calc_loss(self, exp_chi, exp_phi)
        if loss <= 1.0*10**(-6): return True #we have matched!
        else: return ((abs(self.chi) > self.max_chi) or \
                     (abs(self.phi) > self.max_phi) or \
                     (abs(action[-1]) > self.max_two_theta))
        
    #------------------------ ADDED METHODS -----------------------------------

    def init_ub():
        #Prompt user for parameters and reflections
        a = float(input("Please input the length of (or an estimate of the length of) the a axis: "))
        b = float(input("Please input the length of (or an estimate of the length of) the b axis: "))
        c = float(input("Please input the length of (or an estimate of the length of) the c axis: "))
        alpha = float(input("Please input the degree measure of (or an estimate of the degree measure of) alpha: "))
        beta = float(input("Please input an estimate or value for the degree measure of beta: "))
        gamma = float(input("Please input an estimate or value for the degree measure of gamma: "))
        
        h1, k1, l1 = input("Please input a first hkl triple, with h, k, and l separated by commas: ")
        h1 = int(h1); k1 = int(k1); l1 = int(l1)
        print (h1, k1, l1)
        omega1, chi1, phi1 = input("Please input the omega, chi, and phi angles used to find that reflection (again separated by commas): ")
        omega1 = float(omega1); chi1 = float(chi1); phi1 = float(phi1)
        
        h2, k2, l2 = input("Please input a second hkl triple: ")
        h2 = int(h2); k2 = int(k2); l2 = int(l2)
        omega2, chi2, phi2 = input("Please input the omega, chi, and phi angles used to find that reflection: ")
        omega2 = float(omega2); chi2 = float(chi2); phi1 = float(phi2)
        
        #Calculate initial value of UB
        ast, bst, cst, alphast, betast, gammast = ub.star(a, b, c, alpha, beta, gamma) #Calculates reciprocal parameters
        Bmat = ub.calcB(ast, bst, cst, alphast, betast, gammast, c, alpha) #Calculates the initial B matrix
        Umat = ub.calcU(h1, k1, l1, h2, k2, l2, omega1, chi1, phi1, omega2, 
                       chi2, phi2, Bmat)
        ub_0 = np.dot(Umat, Bmat)
        
        return ub_0, Umat, chi2, phi2 #at the end of 2 measurements, we're obviously at the second measurement's location
    
    #Chi
    def calc_M(self):
        M = np.array([[math.cos(self.chi), 0, math.sin(self.chi)], [0, 1, 0], [-math.sin(self.chi), 0, math.cos(self.chi)]])
        return M
    
    #Phi
    def calc_N(self):
        N = np.array([[1, 0, 0], [0, math.cos(self.phi), -math.sin(self.phi)], [0, math.sin(self.phi), math.cos(self.phi)]])
        return N
    
    #Calculated expected angular values for a given hkl
    def calc_expected(self, action):
        q = 4*math.pi()/self.wavelength * math.sin(action[-1]/2.0)
        Q_nu = np.dot(self.ubs[-1], action[0:3])
        
        #MNQ_nu = [q, 0, 0]
        phi_expected = [], chi_expected = [] #For testing cases
        try: 
            q1 = Q_nu[0], q2 = Q_nu[1], q3 = Q_nu[2]
            cossqr_phi = 1.0/(1.0 + (q2/q3)**2)
            phi_expected.append(math.degrees(math.acos(math.sqrt(cossqr_phi))))
            phi_expected.append(math.degrees(math.acos(-math.sqrt(cossqr_phi))))
            phi_expected.append(360.0 - math.degrees(math.acos(math.sqrt(cossqrt_phi))))
            phi_expected.append(360.0 - math.degrees(maht.acos(-math.sqrt(cossqrt_phi))))
        except ZeroDivisionError:
            if q2 != 0: phi_expected.append(90); phi_expected.append(270)
            else: #only q1 is nonzero, and by logic it must be q because sin(chi) = 0 
                if q1 == q:
                    ph = 0 #Automatic
                    ch = 0 #must be zero so cos is 1, sin is 0
                    return ph, ch
                else: 
                    print "The UB matrix should not consist of linearly dependent columns"
                    sys.exit()
                    
        for ph in phi_expected:
            try:
                c = (q2**2/q3 + q3)*math.cos(ph)
                sinsqr_chi = 1.0/(1.0 + (q1/c)**2)
                chi_expected.append(math.degrees(math.asin(math.sqrt(sinsqr_chi))))
                chi.expected.append(math.degrees(math.asin(-math.sqrt(sinsqr_chi))))
            except ZeroDivisionError: #q3 is 0 because c is never zero
                #If we are here, q2 is nonzero
                sinsqr_chi = 1.0/(1.0 + (q1/q2)**2)
                chi_expected.append(math.degrees(math.asin(math.sqrt(sinsqr_chi))))
                chi_expected.append(math.degrees(math.asin(math.sqrt(sinsqr_chi))))
                
        #Testing the possible phi, chi values
        exp_phi = 0; exp_chi = 0 #Default
        for ph in phi_expected:
            for ch in ch_expected:
                val = q1*math.cos(ch) + q2*math.sin(ch)*math.sin(ph) + q3*math.sin(ch)*math.cos(ph)
                if abs(q - val) <= 0.000001:
                    exp_phi = ph; exp_chi = ch
                    
        return exp_chi, exp_phi
    
    #Find the loss - the angular difference
    def calc_loss(self, exp_chi, exp_phi):
        return (self.chi - exp_chi)**2 + (self.phi - exp_phi)**2
    
    ##Take information directly from machine (this is filler)
    #def observe_angles(action):
        ##Get measurements from the machine
        
    def observe_angles():
        good = False
        while good == False:
            try:
                self.chi = float(input("At what chi are you currently measuring? "))
                self.phi = float(input("At what phi are you currenlty measuring? "))
                if abs(self.chi) < 90 and abs(self.phi - 180) < 180: good = True
                else: print "Please input valid numerical values (-90 < chi < 90, and 0 < phi < 360)"
            except:
                print "Please input valid numerical values"
                
    #---------------- INCOMPLETE ----------------
    def update_Umat(self, action, observation):
        exp_chi, exp_phi = self.calc_expected(action)
        loss = self.calc_loss(exp_chi, exp_phi)
        if loss <= 1.0*10**(-6): pass #no changes
        else:
            print "Sorry, still figuring out action selection."
            pass
    
    #Happens after each completed action
    #---------------- INCOMPLETE ----------------
    def add_ub(action, observation):
        observations = {'h' : action[0],
                        'k' : action[1],
                        'l' : action[2],
                        'two_theta': action[3],
                        'str_factor' : observation[0],
                        'f_2' : observation[1],
                        'chi' : self.chi,
                        'phi' : self.phi}
        
        pass
    

        