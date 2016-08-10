import numpy as np
import math
import time
import sys
import random as rd

from rllab.envs.box2d.parser import find_body, find_joint
from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env_ub import Box2DEnvUB
from rllab.misc import autoargs
from rllab.spaces.discrete import Discrete
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
#import rllab.algos.ddpg_with_explore as dwe

import ubmatrix as ub
import lattice_calculator_procedural2 as lcp

class UBEnv(Box2DEnvUB, Serializable):
    
    @autoargs.inherit(Box2DEnvUB.__init__)
    def __init__(self, *args, **kwargs):
        """    Constants:
        omega is always 0, and set a constant for background noise
        """
        
        self.Om = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.background = 1.0 #background noise
        
        """    Variables:
        These variables will be read along with the action:
        two_theta: detector's rotation about the z-axis -- assume elastic scattering, so omega is always 0
        theta: the angle at which our neutrons strike the plane
        
        These variables are the two dimensions of our problem
        chi: outer ring's rotation about the x-axis
        phi: rotation of the eulerian cradle, varies between z- and y-axis rotation depending on how much chi rotated
        
        """
        
        self.max_two_theta = 180
        self.max_chi = 90
        self.max_phi = 360
        self.min_chi = -90
        self.min_phi = 0
        self.correct = 0
        #Set up hkl and all actions
        super(UBEnv, self).__init__(self.model_path("UB.xml.mako"),*args, **kwargs)
        
        self.time = time.time() #Time cost
        
        #Two independent bodies
        self.ring = find_body(self.world, "ring") #chi
        self.eu_cradle = find_body(self.world, "eu_cradle") #phi
        self.detector = find_body(self.world, "detector") #theta
        self.pivot = find_joint(self.world, "angular_axis") #pivot that enables angular movement
        Serializable.__init__(self, *args, **kwargs)    
    
    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        
        #True for each episode, not for all training
        self.hkls = [0]; self.ubs = []; self.U_mat = np.zeros(shape=(3,3))        
        
        """      Variables:
        
        time: used to measure the time cost of the experiment
        ubs: a list of ubs, with a new ub calculated after every measurement
        wavelength: the wavelength used in experiment
        h, k, l: Miller indices. Take the place of x, y, z coordinates
        two_theta, omega, chi, phi: machine angles
        """
        
        self.steps = 0
        super(UBEnv, self).setup_spaces()
        f = open(self.fname, 'r')
        line = f.next(); self.pars = []
        while len(line.split()) == 1:
            self.pars.append(float(line.split()[0]))
            line = f.next()
        
        good = False
        self.wavelength = self.pars[6]
        
        """  Conversion Table
        
        Positions
        theta is discrete
        phi, chi is continuous
        
        Velocities are irrelevant, and acceleration does not exist
        """
        self.chis = []; self.phis = []
        self.ring.position = (0,self.ring.position[1])
        self.eu_cradle.position = (self.eu_cradle.position[0],0)
        self.detector.angle = 0
        self.theta = self.detector.angle
        f.close()
        print "Start at (0,0,0)"
        return self.get_current_obs([0,0,0,0]) #get_current_obs must take an action
    
    @overrides
    def forward_dynamics(self, action):
        assert len(action) == 4, "The action doesn't have the right number of dimensions!"
        choice = math.floor(action[0]+0.5)
        lb, ub = np.array([0,-90,0,0]), np.array([1,90,360,len(self.hkl_actions)-1])
        action = np.clip(action, lb, ub)        
        if choice == 0: #discrete
            ind = math.floor(action[3])
            self.hkls.append(int(ind))
            theta = self.calc_theta(self.hkl_actions[ind][0], self.hkl_actions[ind][1], self.hkl_actions[ind][2])
            if self.steps >= 2:
                exp_chi, exp_phi = self.calc_expected()
                self.move(theta, exp_chi, exp_phi)
            else:
                chi = action[1]; phi = action[2]
                self.move(theta, chi, phi)
            
        elif choice == 1: #continuous
            self.hkls.append(self.hkls[-1]) #measuring the same plane again
            if self.steps == 0:
                theta = self.calc_theta(self.hkl_actions[0][0], self.hkl_actions[0][1], self.hkl_actions[0][2])
            else:
                theta = self.calc_theta(self.hkl_actions[self.hkls[-1]][0], self.hkl_actions[self.hkls[-1]][1], self.hkl_actions[self.hkls[-1]][2])
            self.move(theta, action[1], action[2])
            
        else: 
            print "There are only two choices of move types: discrete, or continuous."
            print "Please do not recreate math."
            raise NotImplementedError
        
    @overrides
    def step(self, action, observation):
        """   This is identical to box2d_env_ub's 
        step method, except adds environment-specific methods,
        changes the order of behaviors    """
        
        #forward the state
        action = self._inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.forward_dynamics(action)
        
        self._invalidate_state_caches()
        if self.steps >= 2:
            if self.steps == 2:
                ub_0, U_0, h, k, l = self.init_ub()
                ind = np.where(np.array([h,k,l])==self.hkl_actions[:,0:3])[0][0]
                self.h2 = h; self.k2 = k; self.l2 = l
                self.ubs.append(ub_0)
                self.U_mat = np.array(U_0)
            else: self.add_ub()        
        done = self.is_current_done(action)
        next_obs = self.get_current_obs(action)
        if next_obs[0] > self.background: success = True
        else: success = False
        if success: self.steps += 1
        reward_computer = self.compute_reward(action, next_obs)
        reward_computer.next()
        reward = reward_computer.next()

        return Step(observation=next_obs, reward=reward, done=done)        
        
    
    @overrides
    def compute_reward(self, action, observation):
        yield
        timeCost = (time.time() - self.time)/1800
        if observation[0] >= self.background:
            accuracy = 1.0
        else: accuracy = 0.0
        
        if self.steps > 2:
            exp_chi, exp_phi = self.calc_expected()
            loss = self.calc_loss(exp_chi, exp_phi)
        else:
            loss = 0
        
        reward = accuracy - timeCost - loss/100.0
        yield reward
    
    @overrides
    def is_current_done(self,action):
        choice = math.floor(action[0]+0.5)
        if choice == 0:
            ind = int(math.floor(action[3]-0.00001))
        else:
            ind = self.hkls[-1]
        act = self.hkl_actions[ind]
        
        if self.steps >= 2:
            exp_chi, exp_phi = self.calc_expected()
            loss = self.calc_loss(exp_chi, exp_phi)
            two_theta = self.calc_theta(act[0], act[1], act[2])
        
            if loss <= 1.0*10**(-2):
                self.correct += 1
                if self.correct == 3:
                    print "The final UB matrix for this experiment is:"
                    print self.ubs[-1]
                    return True #we have matched!
                else: 
                    truth = ((abs(self.chis[-1]) > self.max_chi) or \
                         (abs(self.phis[-1]) > self.max_phi) or \
                         (two_theta > self.max_two_theta))
                    if truth:
                        print "The final UB matrix for this experiment is:"
                        print self.ubs[-1]
                        return True
                    else: return False
            else:
                self.correct = 0
                two_theta = self.calc_theta(act[0], act[1], act[2])
                return ((abs(self.chis[-1]) > self.max_chi) or \
                         (abs(self.phis[-1]) > self.max_phi) or \
                         (two_theta > self.max_two_theta))
            
        else: #We haven't taken enough measurements
            two_theta = self.calc_theta(act[0], act[1], act[2])
            return ((abs(self.chis[-1] > self.max_chi)) or \
                         (abs(self.phis[-1]) > self.max_phi) or \
                         (two_theta > self.max_two_theta))
        
    #------------------------ ADDED METHODS -----------------------------------

    def init_ub(self):
        #Parameters are already in self.pars
        self.time = time.time() #In case the scientist ran to get a sandwich after reset
        
        self.h1 = int(self.hkl_actions[self.hkls[0]][0]); self.k1 = int(self.hkl_actions[self.hkls[0]][1]); self.l1 = int(self.hkl_actions[self.hkls[0]][2])
        self.h2 = int(self.hkl_actions[self.hkls[1]][0]); self.k2 = int(self.hkl_actions[self.hkls[1]][1]); self.l2 = int(self.hkl_actions[self.hkls[1]][2])
        
        #Calculate initial value of UB
        ub_0, Umat = self.calc_mats()
        print "The first UB matrix:"
        print ub_0
        
        return ub_0, Umat, self.h2, self.k2, self.l2 #at the end of 2 measurements, we're obviously at the second measurement's location
    
    def calc_mats(self):
        pars = self.pars
        ast, bst, cst, alphast, betast, gammast = ub.star(pars[0], pars[1], pars[2], pars[3], pars[5], pars[4]) #Calculates reciprocal parameters
        Bmat = ub.calcB(ast, bst, cst, alphast, betast, gammast, pars[2], pars[3]) #Calculates the initial B matrix
        print "The B matrix: " + str(Bmat)
        Umat = ub.calcU(self.h1, self.k1, self.l1, self.h2, self.k2, self.l2, 0, self.chis[0], self.phis[0], 0,
                        self.chis[1], self.phis[1], Bmat)
        print "The U matrix: " + str(Umat)
        ub_0 = np.dot(Umat, Bmat) 
        return ub_0, Umat
    
    def calc_theta(self, h, k, l):
        pars = self.pars
        a = pars[0]; b = pars[1]; c = pars[2]
        alpha = math.radians(pars[3]); beta = math.radians(pars[5]); gamma = math.radians(pars[4])
        theta = 15
        if alpha == beta and abs(math.pi/2 - alpha) < 0.0001: #Ortho, tera, hexa, cubic
            if a != b: #ortho
                val = h**2/a**2 + k**2/b**2 + l**2/c**2
            elif a != c: #Tetra, hexa
                if abs(gamma - 2*math.pi/3) <= 0.00001: #Hexa
                    val = 4.0/3 * ((h**2 + h*k + k**2)/a**2) + l**2/c**2
                else: #Tetra
                    assert abs(gamma - math.pi/2) <= 0.00001, "Check your if statements in the determination of lattoice shape"
                    val = (h**2 + k**2)/a**2 + l**2/c**2   
            else: #cubic
                val = (h**2 + k**2 + l**2)/a**2
                
        elif alpha == beta and alpha == gamma and abs(math.pi/2 - alpha) < 0.0001: #Rombo
            assert a == b, "Something is wrong with this crystal. It's rhombohedral but not."
            assert b == c, "Something is wrong with this crystal. It's rhombohedral but not."
            assert a == c, "Something is wrong with this crystal. It's rhombohedral but not."
            
            denom = a**2*(1 - 3*(math.cos(alpha))**2 + 2*(math.cos(alpha))**3)
            val = (h**2 + k**2 + l**2)*math.sin(alpha)**2 + 2*(h*k + k*l + h*l)*((math.cos(alpha))**2 - math.cos(alpha))
            
        else: #mono or tri
            if beta == gamma and abs(math.pi/2 - alpha) < 0.0001: #mono
                val = 1/math.sin(beta)**2 * \
                    (h**2/a**2 + \
                    (k**2 * math.sin(beta)**2)/b**2 + \
                    l**2/c**2 - \
                    2*h*l*math.cos(beta)/(a*c))
            else: #tri
                #The worst equation ever
                V=2*a*b*c*\
                np.sqrt(np.sin((alpha+beta+gamma)/2)*\
                       np.sin((-alpha+beta+gamma)/2)*\
                       np.sin((alpha-beta+gamma)/2)*\
                       np.sin((alpha+beta-gamma)/2))
                
                S11 = (b*c*math.sin(alpha))**2
                S22 = (a*c*math.sin(beta))**2
                S33 = (a*b*math.sin(gamma))**2
                S12 = a*b*c**2 * (math.cos(alpha)*math.cos(beta) - math.cos(gamma))
                S23 = a**2*b*c * (math.cos(beta)*math.cos(gamma) - math.cos(alpha))
                S31 = a*b**2*c * (math.cos(alpha)*math.cos(gamma) - math.cos(beta))
                val = 1/V**2 * (S11*h**2 + S22*k**2 + S33*l**2 + 2*S12*h*k + 2*S23*k*l + 2*S31*h*l)
        
        d = math.sqrt(1/val)
        if self.wavelength/(2*d) > 1: theta = 90.0
        else: theta = math.degrees(math.asin(self.wavelength/(2*d)))        
        return theta
    
    def move_theta(self, theta):
        goal = math.radians(theta)
        print "Theta goal: " + str(goal)
        accel = 0
        
        self.pivot.motorEnabled = True
        if self.detector.angle < goal: self.pivot.motorSpeed = 1e5
        else: self.pivot.motorSpeed = -1e5
        
        self.before_world_step(accel); count = 0
        while (abs(self.detector.angle - goal) > 0.015):
            self.world.Step(
                self.extra_data.timeStep,
                self.extra_data.velocityIterations,
                self.extra_data.positionIterations
            )
            
            if count == 0:
                self.pivot.maxMotorTorque = 1.0
                self.detector.angularVelocity = 0.1
            if count%20 == 0:
                print "Step %d: theta = %f" % (count, self.detector.angle)
            if count == 5000:
                raise Exception ("Print some stuff please.")
            count += 1
            
    def move_chiphi(self, chi, phi):
        goal = [chi,phi]
        print "Chi and phi goal: " + str(goal)
        accel = 0
        
        self.before_world_step(accel); count = 0
        while(abs(self.ring.position[0] - goal[0]) > 0.11 or \
               abs(self.eu_cradle.position[0] - goal[1]) > 0.11):
            if count > 0:
                self.eu_cradle.linearVelocity = (direction[1]/abs(direction[1]) * max(0.02,abs(2*direction[1])), \
                                                self.eu_cradle.linearVelocity[1])
            self.world.Step(
                self.extra_data.timeStep,
                self.extra_data.velocityIterations,
                self.extra_data.positionIterations           
            )
            
            if count == 0:
                displacement = np.array([chi - self.ring.position[0], phi - self.eu_cradle.position[0]])
                direction = displacement/np.linalg.norm(displacement)
                self.ring.linearVelocity = (direction[0]/abs(direction[0]) * max(0.02,abs(2*direction[0])), self.ring.linearVelocity[1])
 
            count += 1
            if count%20 == 0:
                print "Step %d: chi and phi: %f, %f" % (count, \
                                                            self.ring.position[0], \
                                                            self.eu_cradle.position[0])
            if count >= 5000:
                raise Exception ("Print some stuff please.")                 

    #Move
    def move(self, theta, chi, phi):
        self.move_theta(theta)
        self.move_chiphi(chi, phi)
        
        self.theta = math.degrees(self.detector.angle)
        self.chis.append(self.ring.position[0])
        self.phis.append(self.eu_cradle.position[0])     
    
    #Chi
    def calc_M(self):
        M = np.array([[math.cos(self.chis[-1]), 0, math.sin(self.chis[-1])], [0, 1, 0], [-math.sin(self.chis[-1]), 0, math.cos(self.chis[-1])]])
        return M
    
    #Phi
    def calc_N(self):
        N = np.array([[1, 0, 0], [0, math.cos(self.phis[-1]), -math.sin(self.phis[-1])], [0, math.sin(self.phis[-1]), math.cos(self.phis[-1])]])
        return N
    
    #Calculated expected angular values for a given hkl
    def calc_expected(self):
        ind = int(self.hkls[-1])
        action = self.hkl_actions[ind]
        
        q = 4*math.pi/self.wavelength * math.sin(math.radians(self.calc_theta(action[0], action[1], action[2])))
        Q_nu = np.dot(self.ubs[-1], action)
        print "Q_nu: " + str(Q_nu)
        
        #MNQ_nu = [q, 0, 0]
        phi_expected = []; chi_expected = [] #For testing cases
        try: 
            q1 = Q_nu[0]; q2 = Q_nu[1]; q3 = Q_nu[2]
            cossqr_phi = 1.0/(1.0 + (q2/q3)**2)
            phi_expected.append(math.degrees(math.acos(math.sqrt(cossqr_phi))))
            phi_expected.append(math.degrees(math.acos(-math.sqrt(cossqr_phi))))
            phi_expected.append(360.0 - math.degrees(math.acos(math.sqrt(cossqr_phi))))
            phi_expected.append(360.0 - math.degrees(math.acos(-math.sqrt(cossqr_phi))))
        except ZeroDivisionError:
            if q2 != 0: phi_expected.append(90); phi_expected.append(270)
            else: #only q1 is nonzero, and by logic it must be q because sin(chi) = 0 
                if q1 == q:
                    ph = 0 #Automatic
                    ch = 0 #must be zero so cos is 1, sin is 0
                    return ph, ch
                else: 
                    raise Exception("The UB matrix should not consist of linearly dependent columns")
                    
        for ph in phi_expected:
            try:
                c = (q2**2/q3 + q3)*math.cos(math.radians(ph))
                sinsqr_chi = 1.0/(1.0 + (q1/c)**2)
                chi_expected.append(math.degrees(math.asin(math.sqrt(sinsqr_chi))))
                chi_expected.append(math.degrees(math.asin(-math.sqrt(sinsqr_chi))))
            except ZeroDivisionError: #q3 is 0 because c is never zero
                #If we are here, q2 is nonzero
                sinsqr_chi = 1.0/(1.0 + (q1/q2)**2)
                chi_expected.append(math.degrees(math.asin(math.sqrt(sinsqr_chi))))
                chi_expected.append(math.degrees(math.asin(-math.sqrt(sinsqr_chi))))
                
        #Testing the possible phi, chi values
        exp_phi = 0; exp_chi = 0 #Default
        for ph in phi_expected:
            for ch in chi_expected:
                val = q1*math.cos(math.radians(ch)) + q2*math.sin(math.radians(ch))*math.sin(math.radians(ph)) + q3*math.sin(math.radians(ch))*math.cos(math.radians(ph))
                if abs(q - val) <= 0.15:
                    exp_phi = ph; exp_chi = ch
        
        print "Expected chi and phi values for two theta = %d: %d, %d" % ((2*self.theta), exp_chi, exp_phi)
        return exp_chi, exp_phi
    
    #Find the loss - the angular difference
    def calc_loss(self, exp_chi, exp_phi):
        return (self.chis[-1] - exp_chi)**2 + (self.phis[-1] - exp_phi)**2
                
    def update_Umat(self):
        exp_chi, exp_phi = self.calc_expected()
        loss = self.calc_loss(exp_chi, exp_phi)
        
        if loss <= .75: return self.U_mat
        else:        
            #Choose a previous index to change, among those with very similar 2*theta
            #Random for now                
            ind = np.where(np.array([self.h2, self.k2, self.l2])==self.hkl_actions[:,0:3])[0][0]
            action = self.hkl_actions[ind]
            two_theta = 2*self.calc_theta(action[0], action[1], action[2])
            
            possible = []; i = 0
            for i in range(len(self.hkl_actions)):    
                if abs(self.theta*2 - two_theta) <= 0.5:
                    print "new h2, k2, and l2"
                    print self.h2, self.k2, self.l2
                    if self.hkl_actions[i][0] != self.h2 and self.hkl_actions[i][1] != self.k2 and self.hkl_actions[i][2] != self.l2:
                        possible.append(ind+i)
            
            if possible:
                choice = rd.randint(0, len(possible))
                new2 = self.hkl_actions[choice][0:3]
                self.h2 = new2[0]; self.l2 = new2[1]; self.k2 = new2[2]
                
                #Update
                _, Umat = self.calc_mats(self.chis[-1], self.phis[-1])
                return Umat
            
            return self.U_mat
                
    #Happens after each completed action
    def add_ub(self):
        B_mat = np.dot(np.linalg.inv(self.U_mat), self.ubs[-1]) #U-1UB = B
        self.U_mat = self.update_Umat()
        self.ubs.append(np.dot(self.U_mat, B_mat))
        print "Updated UB:"
        print self.ubs[-1]
