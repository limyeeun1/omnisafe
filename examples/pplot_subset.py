import matplotlib.pyplot as plt
import pandas as pd
import os


LOG_DIR = '/home/limyeeun/src/omnisafe/examples/runs/DDPG-{SafetyRacecarGoal0-v0}/seed-000-2024-01-31-13-52-40'


if __name__ == '__main__':
# Step 1: Read the Data
    
    file_path = '/home/limyeeun/src/omnisafe/examples/runs/DDPG-{SafetyRacecarGoal0-v0}/seed-000-2024-01-23-14-00-43/progress.csv'
    
    data = pd.read_csv(file_path)

    # Step 2: Parse the Data
    # Example: Plotting 'Metrics/EpRet' and 'Metrics/TestEpRet'
    x = data['Train/Epoch']
    y1 = data['Loss/Loss_reward_critic']
    y2 = data['Loss/Loss_pi']
    y3 = data['Value/reward_critic']

    # Step 3: Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label='Loss/Loss_reward_critic',color='dodgerblue')
    plt.plot(x, y2, label='Loss/Loss_pi', color='limegreen')
    plt.plot(x, y3, label='Value/reward_critic', color='violet', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.title('Metrics over Training Epochs')
    plt.legend()
    plt.show()
