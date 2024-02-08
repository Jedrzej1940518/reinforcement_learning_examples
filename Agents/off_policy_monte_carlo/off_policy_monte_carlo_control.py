

import random

from Agents.utils import export_tabular_policy

class OffPolicyMcControl:
    
    #get_state_space -> function that returns every possible state
    def __init__(self, gamma: float, episode_number: int, get_action_space, get_state_space, generate_episode):
            
            self.softness = 0.9
            
            self.gamma = gamma
            self.episode_number = episode_number
            
            #function, thats state s and returns list of possible actions
            self.get_action_space = get_action_space
            
            #function, takes policy and generates an episode based on it with below format:
            # <r0 = 0, s0, a0>, <r1, s1, a1>, ... <rT-1, sT-1, aT-1>, <rT, sT, aT> 
            self.generate_episode = generate_episode
            
            self.C_sa = {}
            self.Q_sa = {}
            self.Pi_s = {}
        
            for s in get_state_space():
                for a in get_action_space(s):
                    self.C_sa[s,a] = 0 
                    self.Q_sa[s,a] = -1000000 #bias, maybe try it out?
                    self.Pi_s[s] = a   #always takes last possible a in space state
        
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
    
    def off_policy_mc(self):
    
        for i in range(self.episode_number):
            
            [_, episode] = self.generate_episode(self.b_policy)
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