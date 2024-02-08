
#todo change to afterstates later

from math import prod
import random
import sys

from numba import jit

from Agents.utils import export_tabular_policy, pick_action, set_policy_e_greedy, set_policy_greedy, e_greedy

class OffPolicyNStepQLearningControl:
    
    def __init__(self, alpha: float, n: int, gamma:float, bias:int, get_starting_state, get_state_space, get_action_space, is_state_terminal, sars):

        #step_size        
        self.alpha = alpha
    
        #takes in state, action, returns reward, state
        self.sars = sars
    
        #target policy MUST be greedy to allow for any exploration....
        self.e_for_target_policy = 0.05
        #states taken in timestep_t
        self.S_t ={}
        self.A_t ={}
        self.R_t ={}
        
        #sigma in time t
        self.sigma_t = {}
        #importance sampling in time t (one step)
        self.p_t = {}

        #n_step_parameter       
        self.n = n
        
        #discout factor
        self.gamma = gamma
        
        #returns starting_state
        self.get_starting_state = get_starting_state
        
        #state_action value, should map state-> action : value dict
        self.Q_s_a = {}
        
        #learned policy, should map state-> action : probabilty dict
        self.pi_s = {}
        
        
        #for argument state returns boolean "is terminal"
        self.is_state_terminal = is_state_terminal
        
        for s in get_state_space():
            action_space = get_action_space(s)

            self.pi_s[s] = { action : 0 for action in action_space}
            
            #bias #sadge
            self.Q_s_a[s] = { action : bias for action in action_space} 
            set_policy_e_greedy(self.pi_s,self.Q_s_a, s, self.e_for_target_policy)

        self.a_0 = 0
        self.t = 0
        self.t_terminal = sys.maxsize
        
        pass
    
     
    #returns part of dicitonary that is bigger or equal to i and smaller than to
    def dict_slice(self, dict, i:int, to:int):
        return {k:v for k, v in dict.items() if k >= i and k < to}
     
    def returns(self, i:int, to:int):
        rewards = self.dict_slice(self.R_t, i, to+1)
        gamma_pows = range(0, to+1 - i)
        discounted_rewards = [(self.gamma ** power) * r for [t_r, r], power in zip(rewards.items(), gamma_pows)]
        return sum(discounted_rewards)
     
    def importance_sampling_ratio(self, i:int, to:int, target_policy, behaviour_policy):
        actions = self.dict_slice(self.A_t, i, to+1)
        states = self.dict_slice(self.S_t, i, to+1)
        action_states = [[action, state] for [t_a, action], [t_s, state] in zip(actions.items(), states.items())]
        
        nominators = [target_policy[state][action] for [action, state] in action_states]
        denominators  = [behaviour_policy[state][action] for [action, state] in action_states]
        
        ratios = [a/b for a, b in zip(nominators, denominators)]
        return prod(ratios)
    #this is shit xD its always 0 
    def importance_sampling_ratio_assuming_e_greedy(self, i:int, to:int, e:float, target_policy):
        actions = self.dict_slice(self.A_t, i, to+1)
        states = self.dict_slice(self.S_t, i, to+1)
        action_states = [[action, state] for [t_a, action], [t_s, state] in zip(actions.items(), states.items())]
    
        nominators = [target_policy[state][action] for [action, state] in action_states]
        denominators  = [target_policy[state][action] * (1-e) + e / len(target_policy[state]) for [action, state] in action_states]
        ratios = [a/b for a, b in zip(nominators, denominators)]
        return prod(ratios)

    def episode_init(self, s_0):

        self.t = 0
        self.t_terminal = sys.maxsize
        
        self.S_t ={}
        self.A_t ={}
        self.R_t ={}
        self.S_t[0] = s_0
        
    
    def t_updated(self):
        return self.t - self.n + 1  
    
    def iteration(self, s, behaviour_policy, e_for_target_policy):
        t = self.t
        s_n = 0
        if t < self.t_terminal:
            a_t = behaviour_policy(s)
            self.A_t[t] = a_t

            [r_n, s_n] = self.sars(s, a_t)
            self.R_t[t+1] = r_n
            self.S_t[t+1] = s_n
        
            if self.is_state_terminal(s_n):
                self.t_terminal = t+1
            else:
                self.A_t[t+1] = behaviour_policy(s_n)
                self.sigma_t[t+1] = 0 # we dont use importance sampling for now self.sigma(t+1)
                self.p_t[t+1] = 1     # so probably fix this later
    
        t_u = self.t_updated()
            
        if t_u >= 0:
            g = 0

            k = min(t+1, self.t_terminal)
                
            while k > t_u:
                if k == self.t_terminal:
                    g = self.R_t[self.t_terminal]
                else:
                    s = self.S_t[k]
                    a = self.A_t[k]
                    v_exp = sum([self.Q_s_a[s][action] * probability for [action, probability] in self.pi_s[s].items()])        
                    g = self.R_t[k] + self.gamma * (self.sigma_t[k] * self.p_t[k] + (1-self.sigma_t[k]) * self.pi_s[s][a]) *(g - self.Q_s_a[s][a]) + self.gamma * v_exp 
                k -= 1
            self.Q_s_a[self.S_t[t_u]][self.A_t[t_u]] += self.alpha * (g - self.Q_s_a[self.S_t[t_u]][self.A_t[t_u]])
            set_policy_e_greedy(self.pi_s, self.Q_s_a, self.S_t[t_u], e_for_target_policy)
        return s_n
    
    def learn_policy(self, max_episodes):
        
        for i in range(max_episodes):
            s = self.get_starting_state()
            self.episode_init(s)
    
            while self.t_updated() != self.t_terminal -1:
                e = 0.9 #1 - (i/max_episodes) * 1/2
                behaviour_policy = lambda s : e_greedy(e, self.pi_s, s) #e greedy behaviour policy
                s = self.iteration(s, behaviour_policy, self.e_for_target_policy) #target policy is also "e greedy"
                self.t += 1
                
                #in case of stuck, please dont happen
                if(self.t >=50000):
                    print("t is very big, stuck!")
                    break
                
            if max_episodes // 10 and i % (max_episodes // 10) == 0 and i != 0:
                print(f'Learning policy {i / max_episodes:.0%} complete.')
        
        #print("Exporting tabular policy") 
        #export_tabular_policy(self.pi_s, "Off_Policy_N_Step_Q_Learning_Control")
        #print("Exporting Q_S_A") 
        #export_tabular_policy(self.Q_s_a, "q_s_a_q_learning")
        callable_policy = lambda s: e_greedy(0, self.pi_s, s) # completly greedy callable polciy 
        return callable_policy