# %%
# Imports
import pickle

# %%
# Read pickle
with open('../models/NEAT/logs/00001.pickle', 'rb') as file:
    logs = pickle.load(file)

logs

# %%
# Test
logs.keys()

# %%
# Test
logs['speeds'][0]