"""
Classes to get the NEAT model running.
There are classes to :
    - Asynchronously capture screens of a window and access the captured screen.
    - Perform a generation step
"""

# Imports
from encodings import normalize_encoding
from xmlrpc.client import Boolean
from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE
import signal
from PIL import ImageGrab, ImageEnhance, Image
import keyboard
import sys
import neat
import time
import os
import cv2
import pickle as pickle

from utils import *

#os.environ["PATH"] = r"d:D:\ProgramData\Anaconda3\envs\trackmanAIenv\Lib\site-packagespywin32_system32;" + os.environ["PATH"]

from threading import Thread, Lock

# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import neat.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')

#os.chdir('./NEAT')
#print(os.getcwd())

if len(sys.argv) < 0:
    print('Not enough arguments.')
    exit()


# Parameters : these should be put in a config file
# image dimensions
w=1280
h=960

n_lines=20
# hyperparams
#threshold = 0.5
no_generations = 1000 #20 for straight with 8sec of max
#max_fitness = 100.0
gamespeed=1
skip_frames=4
kill_time= 3
kill_speed = 10
max_time=12
no_lines = 20 #need to investigate to upscale that
filename_prefix = "models/NEAT/Checkpoints/checkpoint-"
checkpoint = filename_prefix+"256" #None # filename_prefix + "neat-checkpoint-0"
gen=256 #current gen
server_name=f'TMInterface{sys.argv[1]}' if len(sys.argv)>1 else 'TMInterface0'
window_name = 'TrackMania Nations Forever (TMInterface 1.2.0)'
sv = ScreenViewer(n_lines, w, h)


def run_client_gen(client: Client, server_name: str = 'TMInterface0', buffer_size=DEFAULT_SERVER_SIZE):
    """
    Connects to a server with the specified server name and registers the client instance.
    The function closes the connection on SIGBREAK and SIGINT signals and will block
    until the client is deregistered in any way. You can set the buffer size yourself to use for
    the connection, by specifying the buffer_size parameter. Using a custom size requires
    launching TMInterface with the /serversize command line parameter: TMInterface.exe /serversize=size.

    Parameters
    ----------
    client: Client (the client instance to register)
    server_name: str (the server name to connect to, TMInterface0 by default)
    buffer_size: int (the buffer size to use, the default size is defined by tminterface.constants.DEFAULT_SERVER_SIZE)

    Output
    ----------
    L_fit: Array(int) (Fitness for each NN that ran a simulation)
    L_coords: Array(Array([int, int, int])) (each position for the whole run of each NN)
    L_speeds: Array(Array(int)) (each speed for the whole run of each NN)
    L_inputs: Array(Array([int, bool, bool, int])) (each inputs for the whole run of each NN)
    """
    # Instantiate TMInterface object
    iface = TMInterface(server_name, buffer_size)

    def handler(signum, frame):
        iface.close()

    # Close connections
    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    # Register a new client
    iface.register(client)
    while not client.finished:
        time.sleep(0)
    iface.close()

    return client.L_fit,client.L_coords,client.L_speeds,client.L_inputs


def eval_genomes(genomes, config):
    """
    Evaluates every genome (NN) of the generation

    Parameters
    ----------
    genomes: Array(genome) (every NN of the simulation)
    config: neat.Config (config for the neat algorithm)

    Output
    ----------
    None
    """
    # Initialize client
    global gen,kill_time,max_time
    gen+=1
    L_net=[]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        L_net.append(net)
    max_time2=min(max_time,1+0.5*gen)

    # Run gen
    L_fit,L_coords,L_speeds,L_inputs=run_client_gen(GenClient(L_net,max_time2,kill_time, sv, skip_frames, window_name, gamespeed, kill_speed),server_name) #1/6 de sec en plus par génération

    # Update fitness
    for i in range(len(L_fit)):
        genomes[i][1].fitness=L_fit[i]

    # Write generation recap
    filename = 'models/NEAT/Coords/'+str(gen).zfill(5)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_coords,outfile)
    outfile.close()
    filename = 'models/NEAT/Speeds/'+str(gen).zfill(5)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_speeds,outfile)
    outfile.close()

    for i in range(len(L_inputs)):
        filename="models/NEAT/Inputs/"+str(gen).zfill(5)+'_'+str(i).zfill(3)
        l_inputs=L_inputs[i]
        outfile = open(filename,'a')
        for j in range(len(l_inputs)):
            inputs=l_inputs[j]
            time=inputs[0]
            accelerate=inputs[1]
            brake=inputs[2]
            steer=inputs[3]
            if accelerate:
                outfile.write(str(time)+'press up\n')
            if brake:
                outfile.write(str(time)+'press down\n')
            outfile.write(str(time)+'steer '+str(steer).zfill(5)+'\n')
        outfile.close()

def run(config_file, checkpoint=None):
    """
    Run simulation

    Parameters
    ----------
    config_file: str (path to config file)
    checkpoint: str (path to checkpoint)

    Output
    ----------
    None
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    if not checkpoint == None:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=filename_prefix))

    # Run for up to global no generations.
    winner = p.run(eval_genomes, no_generations)
    #winner = p.run(eval_genomes, no_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {0:'r', 1:'l', 2:'u'}
    try:
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    except:
        print('Missing visualize.py file.')

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    # p.run(eval_genomes, 10)


def main():
    """ Main function.
    Launches the simulation.
    """
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, 'models/NEAT/config-feedforward')
    print('Press z to begin.')
    keyboard.wait('z')

    if checkpoint == None:
        for cpt in os.listdir('.'):
            if cpt[:4] == 'neat':
                os.unlink('./'+cpt)

    run(config_path, checkpoint)
    

if __name__ == '__main__':
    main()
