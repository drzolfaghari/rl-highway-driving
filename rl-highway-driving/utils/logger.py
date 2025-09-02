import csv
import os

class Logger:
    def __init__(self, log_dir, filename="training_log.csv"):
        """
        Simple CSV logger for RL training.
        Args:
            log_dir (str): directory where the log file will be saved
            filename (str): name of the CSV file
        """
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)

        # Open file for writing and create CSV writer
        self.file = open(self.filepath, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Write CSV header
        self.writer.writerow(["episode", "reward", "length", "collisions", "actions"])

    def log(self, episode, reward, length, collisions, actions):
        """
        Log one episode of training.
        """
        self.writer.writerow([episode, reward, length, collisions, actions])
        self.file.flush()

    def close(self):
        """
        Close the log file.
        """
        self.file.close()
