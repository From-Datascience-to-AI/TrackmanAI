# TrackManAI

> Neural network driver for Trackmania
> Data Science / AI Project to develop a model that learns how to play Trackmania based on what the user sees

* Authors: Quentin BEHAR and Colin DAVIDSON
* Creation date: 2022-01-01
* see [AUTHORS.md](./docs/AUTHORS.md)


## Table of Contents

1. [Intro and goals](#intro-and-goals)
2. [Details](#details)
3. [Initialize virtual environment](#initialize-virtual-environment)
4. [Setup](#setup)
5. [Folders](#folder)
6. [Important Files](#important-files)
7. [CLI](#cli)
8. [Credits](#credits)


## Intro and goals

Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
My goal here is to implement the NEAT algorithm to trackmania and at term to creat an AI that can have human capabilities at beating TMNF.


## Details

This project uses TMInterface to train the AI. The progress of the AI is saved both in speeds and positions and replays inside Trackmania.\
The AI is capable of beating the first checkpoint in a relative short time so I restarted the training with Replay save and human-like inputs.


## Initialize virtual environment

```
python3 -m pip install -r requirements/dev.txt
```

## Setup

To setup this project and get your AI running / training, you **must** have TM Interface installed.
Then, check that the game's name in the option fits the one you are using. You can now launch TM Interface with the correct resolution (default is 1200x980).

Then, use the cli to do whatever you want the AI to do !


## Folders

```bash
.
├── trackmanai          # main module folder
├── requirements/       # python requirements folder
├── tests/              # package test folder
├── scripts/            # scripts using this package
├── docs/               # additionnal documentation
├── notebooks/          # folder containing notebooks for research
```


## Important files

```bash
.
├── TrackmanAI
├── README.md                       # this file
├── .gitignore                      # elements to ignore for git
├── trackmanai/                     # main module folder
│   ├── cli.py                      # main file to execute
```


## CLI

TODO


## Credits

This project is inspired by the work done by https://github.com/AndrejGobeX/TrackMania_AI

I would like to thank Yosh for his help to use TMInterface.
Big thanks to the community!
