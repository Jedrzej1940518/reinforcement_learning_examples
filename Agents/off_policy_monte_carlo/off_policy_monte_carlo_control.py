

import random
import time

from Agents.utils import export_tabular_policy

class OffPolicyMcControl:
    
    #get_state_space -> function that returns every possible state
    def __init__(self, gamma: float, timelimit: int, bias:int, get_starting_state, get_action_space, get_state_space, model_step):
            
            self.softness = 0.9
            
            self.gamma = gamma
            self.timelimit = timelimit
            
            #returns starting state for episode generation
            self.get_starting_state = get_starting_state
            
            #function, thats state s and returns list of possible actions
            self.get_action_space = get_action_space
            
            #function, takes (s,a), returns [state', reward, done, info]
            self.model_step = model_step
            
            self.C_sa = {}
            self.Q_sa = {}
            self.Pi_s = {}
        
            for s in get_state_space():
                for a in get_action_space(s):
                    self.C_sa[s,a] = 0 
                    self.Q_sa[s,a] = bias #bias, maybe try it out?
                    self.Pi_s[s] = a   #always takes last possible a in space state
                    
            self.off_policy_mc()
        
    #returns probability of choosing action using our b_policy
    def b_a(self, action, state):
        
        action_space = self.get_action_space(state)
        p = self.softness / len(action_space)
        
        if action == self.Pi_s[state]:
            return (1 - self.softness) + p
        else:
            return p

    def b_policy(self, state):
        
        if random.random() < self.softness:
            action_space = self.get_action_space(state)
            return random.choice(action_space)
        
        else:
            return self.Pi_s[state]
    
    def generate_episode(self, policy):
        s = self.get_starting_state()
        done = False
        episode = []
        r = 0
        while not done:
            a = policy(s)
            episode.append([r, s, a])
            [s,r,done, _] = self.model_step(s, a)
        
        return episode
        
    
    def off_policy_mc(self):
    
        start_time = time.time()
        
        while time.time() - start_time < self.timelimit:
            
            episode = self.generate_episode(self.b_policy)
            g = 0 #returns 
            w = 1 #weights
            
            for [r_t, s_t, a_t] in reversed(episode):
                g = self.gamma * g + r_t
                c = self.C_sa[s_t, a_t] + w
                q = self.Q_sa[s_t,a_t]
                
                self.C_sa[s_t, a_t] = c + w
                self.Q_sa[s_t,a_t] = q + w / (c) * (g - q)
                
                a_max_val = self.Q_sa[s_t,a_t]
                
                for a in self.get_action_space(s_t):
                    a_val = self.Q_sa[s_t, a]
                    if a_val >= a_max_val:
                        self.Pi_s[s_t] = a
                        a_max_val = a_val
                
                if a_t != self.Pi_s[s_t]:
                    break

                w = w* 1/(self.b_a (a_t, s_t))
    
        #export_tabular_policy(self.Pi_s, "Off Policy Monte Carlo")
        return self.Pi_s
    
    def get_policy(self, state):
        return self.Pi_s[state]