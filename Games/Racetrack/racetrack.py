import copy
from dataclasses import dataclass
import random
import time
import numpy as np
from Agents.debug_utils import export_agent_results

from Agents.off_policy_monte_carlo.off_policy_monte_carlo_control import OffPolicyMcControl
from Agents.off_policy_n_step_Q_learning.off_policy_n_step_Q_learning_control import OffPolicyNStepQLearningControl

from . import exported_grid
from .show_grid import show_grid


grid = [] # np.array(exported_grid.color_grid)
nrows, ncols = 0,0 #exported_grid.nrows, exported_grid.ncols
colors =  {} #{name:index for index, name in enumerate(exported_grid.colors)}

starting_positions = [] #get_starting_positions()
state_space = {} #init_state_space()
action_space = {} #init_action_space()

@dataclass(frozen=True)  # Make the dataclass immutable
class State:
    x: int
    y: int
    v_l: int #velocity left
    v_r: int #velocity right
    v_f: int
    win: bool

@dataclass(frozen=True)  # Make the dataclass immutable
class Action:
    v_l_d: int #velocity left delta
    v_r_d: int 
    v_f_d: int

def is_starting_position(state:State) -> bool:
    x, y = state.x, state.y
    return [y, x] in starting_positions

def get_starting_state(win = False) -> State:
    [y,x] = get_starting_position()
    
    return State(x,y, 0,0,0, win)

def get_starting_positions() -> [int, int]:
    starting_positions = []
    for y, rows in enumerate(grid):
        for x, columns in enumerate(rows):
            if grid[y,x] == colors['red']:
                starting_positions.append([y,x])
            
    return starting_positions

def get_starting_position():
    return random.choice(starting_positions)
 
def next_state(state: State, action:Action) -> State:

    v_l = state.v_l
    v_r = state.v_r
    v_f = state.v_f
        
    if random.random() > 0.1:
        v_l += action.v_l_d
        v_r += action.v_r_d
        v_f += action.v_f_d

    f = v_f
    r = v_r - v_l
    x = state.x
    y = state.y
    
    death = False
    win = False
    
    while f > 0 or abs(r) > 0:
        if f > 0:
            f-=1
            y -= 1 #minus y goes "up"
        if r > 0:
            x += 1
            r-=1
        elif r < 0:
            x -=1
            r+=1
        
        if x > ncols or x < 0 or y > nrows or y < 0:
            death = True
            break
        
        if grid[y, x] == colors['black']:
            death = True
            break
        
        if grid[y, x] == colors['green']:
            win = True
            break
            
    
    if death:
        return get_starting_state(win)

    return State(x,y,v_l, v_r, v_f, win)


def init_state_space():
    
    print("geting state space")
    
    possible_states = []
    possible_tiles = []
    
    for y in range(nrows):
        for x in range(ncols):
            if grid[y,x] == colors['black']:
                continue
            possible_tiles.append([y,x])
             
    for [y,x] in possible_tiles:
        for v_l in range(0, 6):
            for v_r in range(0,6):
                for v_f in range(0,6):
                    win = grid[y,x] == colors['green']
                    possible_states.append(State(x,y,v_l, v_r, v_f, win))
    
    print("finished getting state space")
    return possible_states


def action_legal(delta, state):
    return state+delta <= 5 and state + delta >=0


def get_state_space():
    return state_space

def init_action_space():
   
    actions = {}
    for state in get_state_space():
        state_actions = []
        for l_d in reversed(range(-1, 2)):
            for r_d in reversed(range(-1, 2)):
                for f_d in reversed(range(-1,2)): #first policy is 1,1,1 which is kinda good
                    
                    if not action_legal(l_d, state.v_l):
                        continue
                    if not action_legal(f_d, state.v_f):
                        continue
                    if not action_legal(r_d, state.v_r):
                        continue
                    if l_d == 0 and r_d == 0 and f_d == 0 and is_starting_position(state):
                        continue
                    state_actions.append(Action(l_d, r_d, f_d))
        
        actions[state] = state_actions
        
    return actions


def get_action_space(state:State):
    return action_space[state]

def generate_episode(policy, for_real = False):

    r_s_a = []
    
    s = get_starting_state()

    while True:
        a = policy(s)
        if s.win == True:
            break
        r = -1
        r_s_a.append([r, s, a])
    
        s = next_state(s, a)
        
        if for_real and len(r_s_a) > 30:
            break
    
    r_s_a.append([1, s, a]) #we won
    win = len(r_s_a) <= 30
    return [win, r_s_a]

def is_state_terminal(state: State):
    return state.win


def sars(state: State, action: Action):
    s = next_state(state, action)
    r = 1 if s.win else -1
    return [r,s]

def show_episode(r_s_a):
    grid_cp = np.array(exported_grid.color_grid)
    for [r, s, a] in r_s_a:
        [x,y] = s.x, s.y
        grid_cp[y,x] = colors['blue']
    show_grid(grid_cp)
    

def monte_carlo_main(gamma, episode_number, policy_runs, visualise: bool = False):
    start_time = time.time()

    ofpmc = OffPolicyMcControl(gamma, episode_number, get_action_space, get_state_space, generate_episode)
    policy = ofpmc.off_policy_mc()
    callable_policy = lambda s : policy[s]
    print(f'MonteCarlo: Finished generating policy. Trying optimal policy {policy_runs} times')
    results = []
    for i in range(policy_runs):
        [win, r_s_a] = generate_episode(callable_policy, True)
        results.append([win, len(r_s_a)])
        if visualise:
            show_episode(r_s_a)

    keys = ["gamma","episode_number", "policy_runs"]
    agent_params = {key: locals()[key] for key in keys}
    t = time.time() - start_time
    export_agent_results("MonteCarlo", t, agent_params, results)
    
        

def n_step_q_learning_main(alpha, n, gamma, bias, episode_number, policy_runs, visualise: bool = False):
    start_time = time.time()
    
    opnsqlc = OffPolicyNStepQLearningControl(alpha, n, gamma, bias, get_starting_state, get_state_space, get_action_space, is_state_terminal, sars)
    callable_policy =opnsqlc.learn_policy(episode_number)
    
    print(f'Finished generating policy. Trying optimal policy {policy_runs} times')
    results = []
    for i in range(policy_runs):
        [win, r_s_a] = generate_episode(callable_policy, True)
        results.append([win, len(r_s_a)])
        if visualise and i < 10:
            show_episode(r_s_a)
    
    keys = ["alpha", "n", "gamma", "bias", "episode_number", "policy_runs"]
    agent_params = {key: locals()[key] for key in keys}
    t = time.time() - start_time
    export_agent_results("Q_Learning", t, agent_params, results)
    
def main():
    global grid, nrows, ncols, colors, state_space, action_space, starting_positions
    
    grid = np.array(exported_grid.color_grid)
    nrows, ncols = exported_grid.nrows, exported_grid.ncols
    colors =  {name:index for index, name in enumerate(exported_grid.colors)}

    state_space = init_state_space()
    action_space = init_action_space()
    starting_positions = get_starting_positions()

    #monte_carlo_main(0.9, 9000, 100)

    n_step_q_learning_main(0.9, 10, 0.95, -20, 50, 100, True)
    #n_step_q_learning_main(0.9, 10, 1, -20, 50, 100)
    #n_step_q_learning_main(0.9, 10, 0.95, -50, 50, 100)

if __name__ == "__main__":
    main()
