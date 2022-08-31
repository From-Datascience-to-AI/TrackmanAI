"""
Functions to use to build a NEAT system.
"""

# Imports
import os
import keyboard
import neat
from utils import ScreenViewer
import pickle as pickle
import configparser
import signal
from tminterface.interface import TMInterface
from tminterface.client import Client
from tminterface.constants import DEFAULT_SERVER_SIZE
import time
import numpy as np

# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import neat.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')

#paths
run_config="../models/config.ini"
model_config="../models/NEAT/config-feedforward"
model_dir="../models/NEAT"
checkpoint="../models/NEAT/Checkpoints/checkpoint-83"#None #"../models/NEAT/Checkpoints/checkpoint-0"
filename_prefix="../models/NEAT/Checkpoints/checkpoint-"

#training vars
no_generations=1000

#config
config_file = configparser.ConfigParser()
config_file.read(run_config)

#global vars from config file
#screenshots vars
w = int(config_file['Window']['w'])
h = int(config_file['Window']['h'])
window_name = config_file['Window']['window_name']
window_name2 = config_file['Window']['window_name2']
n_lines = int(config_file['Image']['n_lines'])
try:
    screen_viewer = ScreenViewer(n_lines, w, h,window_name)
except:
    screen_viewer = ScreenViewer(n_lines, w, h,window_name2)

#simulation vars
server_name = config_file['Game']['server_name']
gamespeed = int(config_file['Game']['gamespeed'])
skip_frames = int(config_file['Game']['skip_frames'])
kill_time = int(config_file['Game']['kill_time'])
kill_speed = int(config_file['Game']['kill_speed'])
max_time = 35#int(config_file['Game']['max_time'])

# NEAT Client Gen Class
class GenClient(Client):
    """ Class to perform a generation step using tminterface.
    """
    def __init__(self, L_net,max_time2):
        super(GenClient,self).__init__()
        self.init_step=True
        self.fitness=0
        self.max_time=max_time2*1000
        self.kill_time=kill_time*1000
        self.current_i=0
        self.net=L_net[0]
        self.max_i=len(L_net)
        self.current_step=0
        self.skip_frames=skip_frames
        self.L_coords=[[] for i in range(len(L_net))]
        self.L_speeds=[[] for i in range(len(L_net))]
        self.L_inputs=[[] for i in range(len(L_net))]
        self.L_fit=[0 for i in range(len(L_net))]
        self.L_net=L_net
        self.checkpoint_count=0
        self.time=-9999
        self.ready_max_steps=60
        self.ready_current_steps=0
        self.finished=False
        self.sv=screen_viewer
        self.gamespeed = gamespeed
        self.kill_speed = kill_speed


    def on_registered(self, iface: TMInterface):
        """ A callback that the client has registered to a TMInterface instance.
        """
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        if self.gamespeed!=1.0:
            iface.set_speed(self.gamespeed)
        iface.execute_command("set sim_priority realtime")
        #iface.execute_command("cam 3") #does not work
        iface.give_up()
    

    def on_deregistered(self,iface):
        """ A callback that the client has been deregistered from a TMInterface instance. 
        This can be emitted when the game closes, the client does not respond in the timeout window, 
        or the user manually deregisters the client with the deregister command.
        """
        print(f'deregistered to {iface.server_name}')


    def on_shutdown(self, iface):
        """ A callback that the TMInterface server is shutting down. 
        This is emitted when the game is closed.
        """
        pass


    def on_run_step(self, iface, _time:int):
        """ Called on each “run” step (physics tick). 
        This method will be called only in normal races and not when validating a replay.

        Parameters
        ----------
        iface: TMInterface (the TMInterface object)
        _time: int (the physics tick)

        Output
        ----------
        None
        """
        # Update time
        self.time=_time
        if self.time<=0:
            # Reset state
            self.accelerate=False
            self.brake=False
            self.steer=0
            self.yaw=0
            self.pitch=0
            self.roll=0
        if self.time>=0:
            # Update state
            state=iface.get_simulation_state()
            speed=state.velocity
            yaw_pitch_roll=state.yaw_pitch_roll
            self.yaw=yaw_pitch_roll[0]
            self.pitch=yaw_pitch_roll[1]
            self.roll=yaw_pitch_roll[2]
            self.L_speeds[self.current_i].append(speed)
            self.L_coords[self.current_i].append(state.position)
            speed=sum([abs(speed[i]) for i in range(3)])
            self.speed=speed
            if self.init_step:
                # Initialize step
                self.init_step=False
                self.current_step+=1
                self.L_raycast = self.sv.getScreenIntersect()
                self.inputs=self.L_raycast
                self.inputs.append(speed)
                self.inputs.append(self.yaw)
                self.inputs.append(self.pitch)
                self.inputs.append(self.roll)
                #iface.log(str(self.img[:-4]))
                output = np.array(self.net.activate(self.inputs))
                #inputs
                self.accelerate=output[1]>0
                self.brake= output[2]>0
                steer = output[0]
                self.steer = int(steer*65536)

            else:
                # Update step
                #one screenshot every skip_frame frames
                if self.current_step%self.skip_frames==0:
                    self.L_raycast = self.sv.getScreenIntersect()
                    self.inputs=self.L_raycast
                    self.inputs.append(speed)
                    self.inputs.append(self.yaw)
                    self.inputs.append(self.pitch)
                    self.inputs.append(self.roll)
                    #iface.log(str(self.img[:-4]))
                    output = np.array(self.net.activate(self.inputs))
                    #inputs
                    self.accelerate=output[1]>0
                    self.brake= output[2]>0
                    steer = output[0]
                    self.steer = int(steer*65536)

                self.L_inputs[self.current_i].append([self.time,self.accelerate,self.brake,self.steer])
                iface.set_input_state(accelerate=self.accelerate,brake=self.brake,steer=self.steer)
                #score for moving up to 0.04 per frame
                self.fitness+=self.speed/10000
                self.current_step+=1
                #update nn situation
                #input to nn
                #output from nn
                #execute the output
                if self.time>=self.max_time:
                    #iface.log(str(self.fitness))
                    self.L_fit[self.current_i]=self.fitness
                    self.fitness=0
                    self.current_i+=1

                    if self.current_i<self.max_i:
                        self.net=self.L_net[self.current_i]
                        self.current_step=0
                        self.init_step=True
                        iface.give_up()
                    else:
                        #self.sv.Stop()
                        iface.close()
                        self.finished=True
                else:
                    if self.time>self.kill_time and self.speed<self.kill_speed:
                        #iface.log(str(self.speed))
                        #iface.log(str(self.fitness))
                        
                        self.L_fit[self.current_i]=self.fitness
                        self.fitness=0
                        self.current_i+=1

                        if self.current_i<self.max_i:
                            self.net=self.L_net[self.current_i]
                            self.current_step=0
                            self.init_step=True
                            iface.give_up()
                        else:
                            #self.sv.Stop()
                            iface.close()
                            self.finished=True


    def on_simulation_begin(self, iface):
        """ Called when a new simulation session is started (when validating a replay).
        """
        pass


    def on_simulation_step(self, iface, _time:int):
        """ Called when a new simulation session is ended (when validating a replay).
        """
        pass


    def on_simulation_end(self, iface, result:int):
        """ Called on each simulation step (physics tick). 
        This method will be called only when validating a replay.
        """
        pass


    def on_checkpoint_count_changed(self, iface, current:int, target:int):
        """ Called when the current checkpoint count changed 
        (a new checkpoint has been passed by the vehicle).

        Parameters
        ----------
        iface: TMInterface (the TMInterface object)
        current: int (the current amount of checkpoints passed)
        target: int (the total amount of checkpoints on the map (including finish))

        Output
        ----------
        None
        """
        # Increase this client's fitness
        self.fitness+=10000/(self.time/1000)
        if current==target:#case of a finish
            # High reward based on time
            self.fitness+=100000/(self.time/10000)
            #iface.log(str(self.fitness))
            iface.prevent_simulation_finish() # Prevents the game from stopping the simulation after a finished race
            
            # Update fitness
            self.L_fit[self.current_i]=self.fitness
            self.fitness=0
            self.current_i+=1

            # Update NN to consider
            if self.current_i<self.max_i:
                self.net=self.L_net[self.current_i]
                self.current_step=0
                self.init_step=True
                iface.give_up()
            else:
                #self.sv.Stop()
                iface.close()
                self.finished=True
                
        #add reward to NN
        #choose to stop the run or not
        pass


    def on_lap_count_changed(self, iface, current:int):
        """ Called when the current lap count changed (a new lap has been passed).
        """
        pass


    def on_custom_command(self, iface, time_from: int, time_to: int, command: str, args: list):
        """
        Called when a custom command has been executed by the user.

        Parameters
        ----------
        iface: TMInterface (the TMInterface object)
        time_from: int (if provided by the user, the starting time of the command, otherwise -1)
        time_to: int (if provided by the user, the ending time of the command, otherwise -1)
        command: str (the command name being executed)
        args: list (the argument list provided by the user)

        Output
        ----------
        None
        """
        pass


    def on_client_exception(self, iface, exception: Exception):
        """
        Called when a client exception is thrown. This can happen if opening the shared file fails, or reading from
        it fails.

        Parameters
        ----------
        iface: TMInterface (the TMInterface object)
        exception: Exception (the exception being thrown)

        Output
        ----------
        None
        """
        print(f'[Client] Exception reported: {exception}')
        #iface.register(self)#try


# NEAT Trainer class
class NEATTrainer():
    """ NEAT TM Trainer.
    This class has NEAT-related methods :
        - Genomes evaluation
        - Training run
    """
    def __init__(self):
        self.filename_prefix = model_dir + "/Checkpoints/checkpoint-"
        if checkpoint == None:
            self.gen = 0
        else:
            checkpoint_infos = checkpoint.split('/')[-1].split('-')
            self.gen = int(checkpoint_infos[-1])
        #super().__init__()


    def run_client_gen(self, client: Client, buffer_size=DEFAULT_SERVER_SIZE):
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
            net = neat.nn.RecurrentNetwork.create(genome, config)
            net.reset()
            L_net.append(net)
        max_time2=min(max_time,1+0.5*self.gen)

        #TODO: check if bug
        #case of bug: launch again the generation
        # Run gen
        L_fit,L_coords,L_speeds,L_inputs=self.run_client_gen(GenClient(L_net,max_time2)) #1/6 de sec en plus par génération

        #TODO: ADD superviser call here to change map if good scores
        threshold=10/100
        n_good_scores=0
        for score in L_fit:
            if score>= 50000:
                n_good_scores+=1
        if n_good_scores/len(L_fit)>threshold:
            raise Exception("succes","Threshold meet")


        # Update fitness
        for i in range(len(L_fit)):
            genomes[i][1].fitness=L_fit[i]

        # Write generation recap
        filename = model_dir+'/Coords/'+str(self.gen).zfill(5)+'.pickle'
        outfile = open(filename,'wb')
        pickle.dump(L_coords,outfile)
        outfile.close()
        filename = model_dir+'/Speeds/'+str(self.gen).zfill(5)+'.pickle'
        outfile = open(filename,'wb')
        pickle.dump(L_speeds,outfile)
        outfile.close()
        #HERE: add logs

        for i in range(len(L_inputs)):
            filename=model_dir+"/Inputs/"+str(self.gen).zfill(5)+'_'+str(i).zfill(3)
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
                            model_config)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
        if not checkpoint == None:
            p = neat.Checkpointer.restore_checkpoint(checkpoint)

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


def train_neat():
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


    # Define config variables


    trainer = NEATTrainer()

    # Wait for user's input
    print('Press z to begin.')
    keyboard.wait('z')

    # Run training
    trainer.run(no_generations)


if __name__ == '__main__':
    train_neat()