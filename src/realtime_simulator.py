import pandas as pd
import time
from config import Config

class TransactionStream:
    def __init__(self, data_path=None):
        self.data_path = data_path if data_path else Config.DATA_PATH
        self.df = None

    def load_data(self, n_samples=1000):
        """Loads a sample of data to simulate streaming."""
        # For simulation, we just read a chunk
        self.df = pd.read_csv(self.data_path, nrows=n_samples)
        return self.df

    def stream(self):
        """Yields transactions one by one."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        for index, row in self.df.iterrows():
            yield row.to_dict()
            time.sleep(0.1) # Simulate delay
