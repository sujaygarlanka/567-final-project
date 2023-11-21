import time
import numpy as np

class Timer():
    def __init__(self):
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def stop(self, stdout):
        elapsed =  time.time() - self.start_time
        if stdout is not None:
            print(f"{stdout} | {elapsed}")
        else:
            return elapsed
        
def get_scalar_values(x):
    return np.array([o.val for o in x.flatten()]).reshape(x.shape)