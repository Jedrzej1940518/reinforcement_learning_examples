import os



def export_agent_results(agent_name, time, agent_params, win_reward):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f'{agent_name}_results.txt')
    
    with open(file_path, 'a') as file:
        
        file.write(f'\n#{agent_name} - {time:.2f} ---> {agent_params} policy results #\n')
        wins = sum([win for win, _ in win_reward]) 
        win_probability = wins / len(win_reward)
        wins = wins if wins else 1
        mean_reward = sum([reward for win, reward in win_reward if win]) / wins
        file.write(f'win probability {win_probability:.2f}, mean reward when win {mean_reward}\n')