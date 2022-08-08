# %%
# Imports
import configparser

# %%
# Read config
config_file = configparser.ConfigParser()
config_file.read("../models/config.ini")
config_file['Window']['window_name']

# %%
# Read config (second test)
with open('../models/config.ini', 'r') as f:
    settings = f.read()
settings


# %%
# Use config
config_file = configparser.ConfigParser()
config_file.read("../models/config.ini")
w, h = config_file['Window']['w'], config_file['Window']['h']


# %%
# Test parse checkpoint
checkpoint = "models/NEAT/Checkpoints/checkpoint-256"
checkpoint.split('/')[-1].split('-')[-1]