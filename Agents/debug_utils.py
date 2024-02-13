import os



def export_agent_results(agent_name, time, results, agent_params, threads = 1):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f'{agent_name}_results.txt')
    
    with open(file_path, 'a') as file:
  
        file.write(f'\n#{agent_name} threads:{threads}- {time:.2f} ---> {agent_params} policy results #\n')
        wins = sum([win for win, _ in results]) 
        win_probability = wins / len(results)
        wins = wins if wins else 1
        mean_reward = sum([reward for win, reward in results if win]) / wins
        file.write(f'win probability {win_probability:.2f}, mean reward when win {mean_reward}\n')