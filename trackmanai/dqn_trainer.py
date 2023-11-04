"""
Functions to use to build a NEAT system.
"""

# Imports
import time
import keyboard
from utils import *
import pickle as pickle
import configparser
import signal
import tensorflow as tf
from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE


# DQN Client Gen Class
class GenClient(Client):
    """ Class to perform a generation step using tminterface.
    """
    def __init__(self, L_net, max_time, kill_time, sv, skip_frames, wn, gs, ks):
        super(GenClient,self).__init__()
        self.init_step=True
        self.fitness=0
        self.max_time=max_time*1000
        self.kill_time=kill_time*1000
        self.current_i=0
        self.current_step=0
        self.skip_frames=skip_frames
        self.L_coords=[]
        self.L_speeds=[]
        self.L_inputs=[]
        self.L_net = L_net
        self.checkpoint_count=0
        self.time=-9999
        self.ready_max_steps=60
        self.ready_current_steps=0
        self.finished=False
        self.sv=sv
        self.sv.getHWND(wn)
        self.gamespeed = gs
        self.kill_speed = ks


    def on_registered(self, iface: TMInterface):
        """ A callback that the client has registered to a TMInterface instance.
        """
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        if self.gamespeed!=1.0:
            iface.set_speed(self.gamespeed)
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
        # TODO : define state saving accurately for backward propagation
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
            self.L_speeds.append(speed)
            self.L_coords.append(state.position)
            speed=sum([abs(speed[i]) for i in range(3)])
            self.speed=speed
            if self.init_step:
                # Initialize step
                self.init_step=False
                self.current_step+=1
                self.im=self.sv.getScreenImg() #try
                self.L_pix_lines=self.sv.L_pix_lines
                self.L_raycast = get_raycast(self.im,self.L_pix_lines)
                self.inputs=self.L_raycast
                self.inputs.append(speed)
                self.inputs.append(self.yaw)
                self.inputs.append(self.pitch)
                self.inputs.append(self.roll)
                # Get outputs
                inputs = tf.convert_to_tensor(self.inputs, dtype=tf.float32)
                inputs = tf.reshape(inputs, shape=(1, 24))
                output = self.L_net(inputs).numpy()
                # Inputs
                self.accelerate = output[0][1] > 0
                self.brake = output[0][2] > 0
                steer = output[0][0]
                self.steer = int(steer*65536)
            else:
                # Update step
                # one screenshot every skip_frame frames
                if self.current_step%self.skip_frames==0:
                    self.im=self.sv.getScreenImg()
                    self.L_pix_lines=self.sv.L_pix_lines
                    self.L_raycast = get_raycast(self.im,self.L_pix_lines)
                    self.inputs=self.L_raycast
                    self.inputs.append(speed)
                    self.inputs.append(self.yaw)
                    self.inputs.append(self.pitch)
                    self.inputs.append(self.roll)
                    # Get outputs
                    inputs = tf.convert_to_tensor(self.inputs, dtype=tf.float32)
                    inputs = tf.reshape(inputs, shape=(1, 24))
                    output = self.L_net(inputs).numpy()
                    # Inputs
                    self.accelerate = output[0][1] > 0
                    self.brake = output[0][2] > 0
                    steer = output[0][0]
                    self.steer = int(steer*65536)
                # Define inputs to play
                self.L_inputs.append([self.time,self.accelerate,self.brake,self.steer])
                iface.set_input_state(accelerate=self.accelerate,brake=self.brake,steer=self.steer)
                # Score for moving up to 0.04 per frame
                self.fitness+=self.speed/10000
                self.current_step+=1
                #update nn situation
                #input to nn
                #output from nn
                #execute the output
                # TODO : define final states
                if self.time>=self.max_time:
                    # Save fitness
                    
                    # New try
                    iface.close()
                    self.finished=True
                else:
                    if self.time>self.kill_time and self.speed<self.kill_speed:
                        # Save fitness

                        # New try
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
            
            # TODO : define what to do after finish
            # Save fitness

            # New try
            iface.close()
            self.finished=True


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


class L_net(tf.keras.Model):
    """ Standard model
    """
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint == None:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(24,), dtype='float32', name='input_layer'),
                tf.keras.layers.Dense(32, activation='relu', name='hidden_layer'),
                tf.keras.layers.Dropout(0.2, name='dropout_layer'),
                tf.keras.layers.Dense(3, activation='linear', name='output_layer')
            ])
        else:
            self.model = tf.keras.models.load_model(checkpoint)
    
    def call(self, x):
        return self.model(x)


# DQN Trainer class
class DQNTrainer(TMTrainer):
    """ DQN TM Trainer.
    """
    def __init__(self, model_dir, model_config, w, h, window_name, 
    n_lines, server_name, gamespeed, skip_frames, 
    kill_time, kill_speed, max_time, no_lines, 
    screen_viewer, checkpoint=None):
        super().__init__(model_dir, model_config, w, h, window_name, 
        n_lines, server_name, gamespeed, skip_frames, 
        kill_time, kill_speed, max_time, no_lines,
        screen_viewer, checkpoint)


    def run(self, no_generations=1000, server_name: str = 'TMInterface0', buffer_size=DEFAULT_SERVER_SIZE):
        """
        Run simulation

        Parameters
        ----------
        config_file: str (path to config file)
        checkpoint: str (path to checkpoint)
        no_generations: int (number of generations)
        server_name: str (the server name to connect to, TMInterface0 by default)
        buffer_size: int (the buffer size to use, the default size is defined by tminterface.constants.DEFAULT_SERVER_SIZE)

        Output
        ----------
        None
        """
        # Create / load model
        model = L_net(self.checkpoint)

        # Create client
        client = GenClient(model, self.max_time, self.kill_time, self.screen_viewer, self.skip_frames, self.window_name, self.gamespeed, self.kill_speed)

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


def train_dqn(model_dir="./models/NEAT",
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
    pass


if __name__ == '__main__':
    train_dqn(model_dir="../models/NEAT",
    run_config="../models/config.ini",
    model_config="../models/NEAT/config-feedforward",
    checkpoint="../models/NEAT/Checkpoints/checkpoint-0",
    no_generations=1000)