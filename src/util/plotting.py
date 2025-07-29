import matplotlib.pyplot as plt

def plot_training_progress(rewards, td_errors, plot_path="training_plot.png", agent_name="Agent"):
    """
    Plots and saves the training progress (rewards and TD errors).
    Args:
        rewards (list or array): Total rewards per episode.
        td_errors (list or array): Average TD error per episode.
        plot_path (str): Path to save the plot image.
        agent_name (str): Name of the agent for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Total Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg TD Error', color=color)
    ax2.plot(td_errors, color=color, label='Avg TD Error')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"{agent_name} Training Progress")
    plt.savefig(plot_path)
    plt.close(fig)
