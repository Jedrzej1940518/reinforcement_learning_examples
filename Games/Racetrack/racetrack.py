from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import copy
from dataclasses import dataclass
import os
import random
import string
import time
from typing import Any
import numpy as np
from Agents.debug_utils import export_agent_results
from Agents.monte_carlo_tree_search.MonteCarloTreeSearch import MonteCarloTreeSearch

from Agents.off_policy_monte_carlo.off_policy_monte_carlo_control import OffPolicyMcControl
from Agents.off_policy_n_step_Q_learning.off_policy_n_step_Q_learning_control import OffPolicyNStepQLearningControl

from . import exported_grid
from .show_grid import show_grid


grid = [] # np.array(exported_grid.color_grid)
nrows, ncols = 0,0 #exported_grid.nrows, exported_grid.ncols
colors =  {} #{name:index for index, name in enumerate(exported_grid.colors)}

starting_positions = [] #initialize_starting_positions()
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

def initialize_starting_positions() -> list[int, int]:
    starting_positions = []
    for y, rows in enumerate(grid):
        for x, columns in enumerate(rows):
            if grid[y,x] == colors['red']:
                starting_positions.append([y,x])
            
    return starting_positions

def get_starting_position():
    return random.choice(starting_positions)
 
def next_state(state: State, action:Action, randomized: bool = True) -> State:

    v_l = state.v_l
    v_r = state.v_r
    v_f = state.v_f
        
    if not randomized or random.random() > 0.1:
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
        
        if x >= ncols or x < 0 or y >= nrows or y < 0:
            death = True
            break
        
        if grid[y, x] == colors['black']:
            death = True
            break
        
        if grid[y, x] == colors['green']:
            win = True
            break
            
    
    if death:
        return get_starting_state()

    return State(x,y,v_l, v_r, v_f, win)


def init_state_space():
     
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
    
    return possible_states


def action_legal(delta, state):
    return state+delta <= 5 and state + delta >=0

def action_leads_to_zero_speed(state: State, action: Action):
    state_after_action = State(state.x, state.y, state.v_l + action.v_l_d, state.v_r + action.v_r_d, state.v_f + action.v_f_d, state.win)
    return (state_after_action.v_l == state_after_action.v_r and state_after_action.v_f == 0) or state_after_action.v_f < 0

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
                    if action_leads_to_zero_speed(state, Action(l_d, r_d, f_d)):
                        continue
                    if f_d <= 0 and is_starting_position(state):
                        continue

                    state_actions.append(Action(l_d, r_d, f_d))
        
        actions[state] = state_actions
        
    return actions


def get_action_space(state:State):
    return action_space[state]

def generate_episode(policy, randomized = False, for_real = False, step_by_step_visualise = False):

    episode_too_long = 30
    
    r_s_a = []
    
    s = get_starting_state()

    while True:
        a = policy(s)
        if s.win == True:
            break
        r = -1
        r_s_a.append([r, s, a])
        
        if step_by_step_visualise:
            show_episode(r_s_a)

        s = next_state(s, a, randomized)
        
        if for_real and len(r_s_a) > episode_too_long:
            break
    
    r_s_a.append([1, s, a]) #we won
    win = len(r_s_a) <= episode_too_long
    return [win, r_s_a]


def show_episode(r_s_a):
    grid_cp = np.array(exported_grid.color_grid)
    for [r, s, a] in r_s_a:
        [x,y] = s.x, s.y
        grid_cp[y,x] = colors['blue']
    show_grid(grid_cp)
    
    
def hash_state(state: State) -> str:
    # Normalize ranges
    v_l_normalized = state.v_l + 5
    v_r_normalized = state.v_r + 5
    v_f_normalized = state.v_f + 5

    # Encode win as 0 or 1
    win_encoded = int(state.win)
    
    # Concatenate with delimiter
    hash_parts = [
        str(state.x).zfill(2),  # Ensure 2 digits
        str(state.y).zfill(2),  # Ensure 2 digits
        str(v_l_normalized).zfill(2),
        str(v_r_normalized).zfill(2),
        str(v_f_normalized).zfill(2),
        str(win_encoded)
    ]
    hash_str = '-'.join(hash_parts)
    
    # Optional: Convert to a higher base if desired
    # This step is skipped in this example for simplicity
    
    return hash_str
    

def model_step(state: State, action: Action) -> tuple[State, int, bool, Any]:
    s = next_state(state,action, False)
    r = 1 if s.win else -1
    return [s, r, s.win, None]


def initialization():
    global grid, nrows, ncols, colors, state_space, action_space, starting_positions
    
    grid = np.array(exported_grid.color_grid)
    nrows, ncols = exported_grid.nrows, exported_grid.ncols
    colors =  {name:index for index, name in enumerate(exported_grid.colors)}

    state_space = init_state_space()
    action_space = init_action_space()
    starting_positions = initialize_starting_positions()

def run_agent(agent, policy_runs, visualise_func, hiperparams, env_functions):

    ag = agent(**hiperparams, **env_functions)
    callable_policy = lambda s: ag.get_policy(s)
    results = []
    for i in range(policy_runs):
        [win, r_s_a] = generate_episode(callable_policy, randomized=False, for_real = True)
        results.append([win, len(r_s_a)])
        if visualise_func:
            visualise_func(r_s_a)

    return results
    
def time_agent(agent, initialize_env, policy_runs, visualise_func, hiperparams, env_functions):
    
    if initialize_env:
        initialize_env()
        
    print("Agent:",agent.__name__," Pid:", os.getpid(), ", args: ", hiperparams)
    start_time = time.time()
    results = run_agent(agent, policy_runs, visualise_func, hiperparams, env_functions)
    t = time.time() - start_time
    return [t, results]
   
def multithreaded_run(threads, export_results, func, **kwargs):
    results = []
    times = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(func, **kwargs) for _ in range(threads)]
        for future in futures:
            [t, result] = future.result()
            results.append(*result)
            times.append(t)

    if export_results:
        export_agent_results(kwargs["agent"].__name__, max(times), results, kwargs["hiperparams"], threads)

def mcts_main():
    hiperparams = {"gamma": 0.95, "bias": -20, "exploration_degree": 0.2, 
                "tree_policy_depth": 2, "timelimit": 15}
    env_functions = {"hash_state": hash_state, "get_action_space": get_action_space, "model_step": model_step}

    kwargs = {  "agent":MonteCarloTreeSearch,
                "initialize_env":initialization, 
                "policy_runs":1,
                "visualise_func":None, 
                "hiperparams":hiperparams, 
                "env_functions":env_functions}

    multithreaded_run(10, True, time_agent, **kwargs)
    
    hiperparams["bias"] = -25
    
    multithreaded_run(10, True, time_agent, **kwargs)
    
    hiperparams["bias"] = -20
    hiperparams["tree_policy_depth"] = 5
    
    multithreaded_run(10, True, time_agent, **kwargs)
    
    hiperparams["tree_policy_depth"] = 2
    hiperparams["exploration_degree"] = 0.1
    
    multithreaded_run(10, True, time_agent, **kwargs)
    hiperparams["tree_policy_depth"] = 2
    hiperparams["exploration_degree"] = 0.3
    multithreaded_run(10, True, time_agent, **kwargs)
    hiperparams["tree_policy_depth"] = 5
    multithreaded_run(10, True, time_agent, **kwargs)
    
def mc_main():
    
    hiperparams = {"gamma": 0.95, "bias":-100000, "timelimit": 10}
    env_functions = {"get_state_space": get_state_space, "get_starting_state":get_starting_state, "get_action_space": get_action_space, "model_step": model_step}

    kwargs = {  "agent":OffPolicyMcControl,
                "initialize_env":initialization, 
                "policy_runs":1,
                "visualise_func":show_episode, 
                "hiperparams":hiperparams, 
                "env_functions":env_functions}

    multithreaded_run(10, True, time_agent, **kwargs)
    
    hiperparams["timelimit"] = 30
    
    multithreaded_run(10, True, time_agent, **kwargs)


def q_learning_main():
   
    hiperparams = { "alpha":0.9, "n": 10, "gamma":0.95, "timelimit": 600, "bias": -20}
    env_functions = {"get_state_space": get_state_space, "get_starting_state":get_starting_state, "get_action_space": get_action_space, "model_step": model_step}

    kwargs = {  "agent":OffPolicyNStepQLearningControl,
                "initialize_env":initialization, 
                "policy_runs":1,
                "visualise_func":show_episode, 
                "hiperparams":hiperparams, 
                "env_functions":env_functions}

    multithreaded_run(5, True, time_agent, **kwargs)
        

def main():
     mcts_main()
     #mc_main()
     #q_learning_main()

if __name__ == "__main__":
    main()