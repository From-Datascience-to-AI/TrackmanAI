"""
Functions to use to build a NEAT system.
"""

# Imports
import os
import keyboard
import neat
from utils import *
import pickle as pickle
import configparser

# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import neat.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')


# Create NEAT Trainer class
class NEATTrainer(TMTrainer):
    """ NEAT TM Trainer.
    This class has NEAT-related methods :
        - Genomes evaluation
        - Training run
    """
    def __init__(self, model_dir, model_config, w, h, window_name, 
    n_lines, server_name, gamespeed, skip_frames, 
    kill_time, kill_speed, max_time, no_lines, 
    screen_viewer, checkpoint=None):
        super().__init__(model_dir, model_config, w, h, window_name, 
        n_lines, server_name, gamespeed, skip_frames, 
        kill_time, kill_speed, max_time, no_lines,
        screen_viewer, checkpoint)


    def eval_genomes(self, genomes, config):
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
        self.gen+=1
        L_net=[]
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            L_net.append(net)
        max_time2=min(self.max_time,1+0.5*self.gen)

        # Run gen
        L_fit,L_coords,L_speeds,L_inputs=run_client_gen(GenClient(L_net,max_time2,self.kill_time, self.screen_viewer, self.skip_frames, self.window_name, self.gamespeed, self.kill_speed),self.server_name) #1/6 de sec en plus par génération

        # Update fitness
        for i in range(len(L_fit)):
            genomes[i][1].fitness=L_fit[i]

        # Write generation recap
        filename = self.model_dir+'/Coords/'+str(self.gen).zfill(5)+'.pickle'
        outfile = open(filename,'wb')
        pickle.dump(L_coords,outfile)
        outfile.close()
        filename = self.model_dir+'/Speeds/'+str(self.gen).zfill(5)+'.pickle'
        outfile = open(filename,'wb')
        pickle.dump(L_speeds,outfile)
        outfile.close()

        for i in range(len(L_inputs)):
            filename=self.model_dir+"/Inputs/"+str(self.gen).zfill(5)+'_'+str(i).zfill(3)
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


    def run(self, no_generations=1000):
        """
        Run simulation

        Parameters
        ----------
        config_file: str (path to config file)
        checkpoint: str (path to checkpoint)
        no_generations: int (number of generations)

        Output
        ----------
        None
        """
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            self.model_config)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
        if not self.checkpoint == None:
            p = neat.Checkpointer.restore_checkpoint(self.checkpoint)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=self.filename_prefix))

        # Run for up to global no generations.
        winner = p.run(self.eval_genomes, no_generations)
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


def train_neat(model_dir="./models/NEAT",
    run_config="./models/config.ini",
    model_config="./models/NEAT/config-feedforward",
    checkpoint="./models/NEAT/Checkpoints/checkpoint-0",
    no_generations=1000):
    """
    Main function.

    Parameters
    ----------
    model_dir: str (path to model)
    run_config: str (path to run configuration)
    model_config: str (path to model configuration)
    checkpoint: str (name of the checkpoint to load)
    no_generations: int (number of generations to run)

    Output
    ----------
    None
    """
    # Read run config file
    config_file = configparser.ConfigParser()
    config_file.read(run_config)

    # Define config variables
    w, h, window_name = int(config_file['Window']['w']), int(config_file['Window']['h']), config_file['Window']['window_name']
    n_lines = int(config_file['Image']['n_lines'])
    server_name, gamespeed, skip_frames = config_file['Game']['server_name'], int(config_file['Game']['gamespeed']), int(config_file['Game']['skip_frames'])
    kill_time, kill_speed = int(config_file['Game']['kill_time']), int(config_file['Game']['kill_speed'])
    max_time, no_lines = int(config_file['Game']['max_time']), int(config_file['Game']['no_lines'])
    screen_viewer = ScreenViewer(n_lines, w, h)

    trainer = NEATTrainer(model_dir, model_config, w, h, window_name, 
    n_lines, server_name, gamespeed, skip_frames,
    kill_time, kill_speed, max_time, no_lines,
    screen_viewer, checkpoint)

    # Wait for user's input
    print('Press z to begin.')
    keyboard.wait('z')

    # Run training
    trainer.run(no_generations)


if __name__ == '__main__':
    train_neat(run_config="../models/config.ini",
    model_config="../models/NEAT/config-feedforward",
    checkpoint="../models/NEAT/Checkpoints/checkpoint-0",
    no_generations=1000)