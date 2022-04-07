# This project comes from the work done by https://github.com/AndrejGobeX/TrackMania_AI

I modified this project to train a RL AI in TrackMania Nations Forever using the NEAT algorithm.
The functions that remains are the screen computing and part of the implementation of the NEAT algorithm.

# TrackManAI
Neural network driver for Trackmania

## Intro and goals
Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
My goal here is to implement the NEAT algorithm to trackmania and at term to creat an AI that can have human capabilities at beating TMNF.\

## Details
This project uses TMInterface to train the AI. The progress of the AI is saved both in speeds and positions and replays inside Trackmania.\
The AI is capable of beating the first checkpoint in a relative short time so I restarted the training with Replay save and human-like inputs.

## Contents
All the scripts contain instructions for arguments, running, etc. so read them before playing.
| Filename | Brief description |
| -------- | ----------------- |
| NEAT_Trainer.py | Neuroevolution trainer |

## Setup

## Packages used and credits
I would like to thank Yosh for his help to use TMInterface.
Big thanks to the community!\
The latest versions are used for all packages (31.10.2021.):
* TMInterface - communication with TMInterface
* python-pillow/Pillow - image preprocessing
* boppreh/keyboard - keyboard API
* numpy/numpy - array manipulation
* opencv/opencv - image preprocessing
