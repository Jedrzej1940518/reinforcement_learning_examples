import random

def export_tabular_policy(p_s_a, name):

    with open(f'{name}.py', 'w') as file:

        file.write(f"# {name} policy #\n")

        file.write('policy = {\n')
        for [state, actions] in p_s_a.items():
            file.write('\t' + str(state) + ':{\n')
            for [action, value] in actions.items():
                file.write('\t\t' + str(action) + ':' + str(value) + ',\n')
            file.write('},')
        file.write('}\n\n')

def set_policy_e_greedy(p_s_a, Q_s_a, s, e):
    optimal_action = max(Q_s_a[s], key=Q_s_a[s].get)
    min_prob = e / len(Q_s_a[s])
    for [action, _] in p_s_a[s].items():
        p_s_a[s][action] = min_prob
    p_s_a[s][optimal_action] = min_prob + 1-e

def set_policy_greedy(p_s_a, Q_s_a, s):
    optimal_action = max(Q_s_a[s], key=Q_s_a[s].get)
    for [action, _] in p_s_a[s].items():
        p_s_a[s][action] = 0
    p_s_a[s][optimal_action] = 1
  
def e_greedy_w_policy(e, policy, state):
        
    if random.random() < e:
        i = list(policy[state].keys())
        return random.choice(i)
    else:
        return max(policy[state], key=policy[state].get)
        
            
def pick_action(policy, state):
    r = random.random()
    cum_probability = 0
    def_action = -1

    for [action, probability] in policy[state].items():
        def_action = action
        cum_probability += probability
            
        if cum_probability > r:
                return action
        
    if def_action == -1:
        print("error wariacie")
            
    return def_action

def e_greedy(e, action_values):
        
    if random.random() < e:
        [action, value] = random.choice(action_values)
        return action
    else:
         [action, value] = max(action_values, key=lambda x: x[1])
         return action