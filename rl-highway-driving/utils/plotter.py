import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file, save_path="plot.png"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 8))

    # Rewards
    plt.subplot(2,2,1)
    plt.plot(df["episode"], df["reward"], label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()

    # Episode Length
    plt.subplot(2,2,2)
    plt.plot(df["episode"], df["length"], label="Length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Lengths")
    plt.legend()

    # Collisions
    plt.subplot(2,2,3)
    plt.plot(df["episode"], df["collisions"], label="Collisions", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Collisions")
    plt.title("Collisions per Episode")
    plt.legend()

    # Actions
    plt.subplot(2,2,4)
    plt.plot(df["episode"], df["actions"], label="Actions", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Actions")
    plt.title("Actions per Episode")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
