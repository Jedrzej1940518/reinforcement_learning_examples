from dataclasses import dataclass
from math import log, sqrt
import random
import time
from typing import Any
import numpy as np
from collections import defaultdict

from treelib import Node, Tree

from Agents.utils import e_greedy

@dataclass()
class ActionNode:
    action: Any
    value: np.float32
    n_t: int = 0 #number of times action was selected prior to time T

@dataclass()
class StateNode:
    state: Any

#TODO optimizations such as:
#RAVE, eary stopping, pruning...

class MonteCarloTreeSearch:


    def __init__(self, gamma, bias, exploration_degree, tree_policy_depth, timelimit, hash_state, get_action_space, model_step):
        
        #takes state S, returns action_space
        self.get_action_space = get_action_space
        
        #takes state S, action A, returns [state', reward, done, info] - this triggers a model of env
        self.model_step = model_step
 
        #action node taken during last decision making - will help us return to proper state node
        self.last_action_node_taken = None
        #action to update
        self.action_nodes_to_update = []
        
        #returns for one iteration
        self.returns = []
    
        self.t = 0
        
        #takes in state, returns STRING, as short as possible please [*] also NO COLLISIONS !!!
        self.hash_state = hash_state
        #Tree of state_nodes -> action_nodes (many) -> state_node
        #each Node (state) shall hold whole action space
        self.tree = Tree()
        self.starting_gamma = gamma
        self.starting_bias = bias
        self.starting_tree_policy_depth = tree_policy_depth
        self.starting_e = exploration_degree
        self.timelimit = timelimit
        
        pass
    
    #discount factor
    def gamma(self):
        return self.starting_gamma
    
    #bias - default value of action
    def bias(self):
        return self.starting_bias
    
    #default tree policy depth without expansion
    def tree_policy_depth(self):
        return self.starting_tree_policy_depth
    
    def e(self):
        return self.starting_e
    
    def c(self):
        return self.starting_e

    #TODO this is supposed to be ucb for trees, for now idc
    def upper_confidence_bound(self, action_node):
        maximizing_value = np.float32(999999999999)
        if action_node.data.n_t == 0:
            return maximizing_value
        
        return action_node.data.value + self.c() * sqrt(log(self.t+2) / action_node.data.n_t)

    #takes in state_node, returns action_node during selection process 
    def tree_policy(self, state_node):
        action_nodes = self.tree.children(state_node.identifier)
        
        action_values = [[action_node, self.upper_confidence_bound(action_node)] for action_node in action_nodes]
        action_node =  e_greedy(0, action_values)
        action_node.data.n_t +=1

        return action_node
    
    def rollout_policy(self, state):
        actions = self.get_action_space(state)
        action = random.choice(actions)

        return action
    
    def target_policy(self, state_node):
        action_nodes = self.tree.children(state_node.identifier)
        action_values = [[action_node, action_node.data.value] for action_node in action_nodes]
        #completly greedy policy
        return e_greedy(0, action_values)
    
    def add_action_nodes(self, state_node):
        
        if not state_node.is_leaf(): #if state already has children, we dont need to add action nodes
            return
        
        state = state_node.data.state
        action_space = self.get_action_space(state)
        for action in action_space:
            self.tree.create_node(parent=state_node.identifier, data=ActionNode(action, self.bias()))


    def selection(self, state_node):
        
        self.add_action_nodes(state_node)
        state = state_node.data.state
    
        for i in range(self.tree_policy_depth()):

            action_node = self.tree_policy(state_node)
            self.action_nodes_to_update.append(action_node)
            action = action_node.data.action
    
            [state, reward, done, _] = self.model_step(state, action)
            self.returns.append(reward)
            #find proper state node under action node            
            #TODO what if this state node had different parent? This state node should have two parents - maybe this is not big deal - 
            state_node = self.tree.get_node(self.hash_state(state))
            
            #if doesnt exist, create
            if state_node == None:
                state_node = self.tree.create_node(identifier=self.hash_state(state), parent=action_node, data=StateNode(state))
            
            if done:
                return [done, state_node]
            #also i think adding action nodes is wrong? too many actions
            self.add_action_nodes(state_node)

        return [False, state_node]

    def expansion(self, state_node):
        return state_node
    
    def simulation(self, state_node):

        #this looks silly but its' according to barto sutton 208 page
        action_node = self.tree_policy(state_node)
        action = action_node.data.action
        [state, reward, done, _] = self.model_step(state_node.data.state, action)
        self.action_nodes_to_update.append(action_node)
        self.returns.append(reward)

        current_discount = self.gamma()      
        #we're not saving rewards or anything here...maybe we could one day
        while not done:
            a = self.rollout_policy(state)
            [state, reward, done, _] = self.model_step(state, a)
            self.returns[-1] += reward * current_discount
            current_discount *= self.gamma()

        return 
    def backup(self):
        g = 0
        for [r, action_node] in zip(reversed(self.returns), reversed(self.action_nodes_to_update)):
            #first r already contains all the already discounted rewards untill termination TODO maybe this is useuless optimization
            g = self.gamma() * g + r 
            action_val = action_node.data.value
            action_node.data.value = action_val + 1/action_node.data.n_t * (g - action_val)
        
        self.returns = []
        self.action_nodes_to_update = []
       
    def iteration(self, state_node):

        [done, state_node] = self.selection(state_node)
        if not done:
            state_node = self.expansion(state_node)
            self.simulation(state_node)
        
        
        self.backup()
        
    def get_current_root(self, state):
        
        state_node = self.tree.get_node(self.tree.root)
        #tree was empty
        if state_node == None:
            state_node = self.tree.create_node(identifier=self.hash_state(state),data=StateNode(state))
            return state_node

        #tree isnt empty, we look for already saved state_node
        state_node = self.tree.get_node(self.hash_state(state))
            
        #we didnt have a proper state_node - we need to create it under last taken action
        if state_node == None:
            state_node = self.tree.create_node(parent=self.last_action_node_taken.identifier, identifier=self.hash_state(state),data=StateNode(state))
            return state_node

        return state_node
    
    def reset_time(self):
        self.t=0
      
    def get_policy(self, state):
        start_time = time.time()
        
        state_node = self.get_current_root(state)

        #we need to iterate at least once
        self.iteration(state_node)

        #TODO fix this shit
        while time.time() - start_time < self.timelimit:
            self.iteration(state_node)

           

        action_node = self.target_policy(state_node)
        
        self.last_action_node_taken = action_node
        self.t +=1
        return action_node.data.action
    
