from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE
import signal
from PIL import ImageGrab, ImageEnhance, Image
import keyboard
import sys
import numpy as np
import neat
import time
import os
import cv2
import win32gui
import DirectKey
import pickle as pickle

KEY_ENTER = 0x1C

# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import neat.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')

os.chdir('./NEAT')
print(os.getcwd())

if len(sys.argv) < 0:
    print('Not enough arguments.')
    exit()

# image dimensions
image_width = 100
image_height = 100

# hyperparams
#threshold = 0.5
no_generations = 1000

#max_fitness = 100.0
gamespeed=1
skip_frames=5
no_seconds = 40
kill_seconds = 3
kill_steps= kill_seconds*60
kill_speed = 20
no_lines = 5
checkpoint = None#"neat-checkpoint-280"
gen=0 #current gen
max_gen=280
server_name=f'TMInterface{sys.argv[1]}' if len(sys.argv)>1 else 'TMInterface0'

def save_replay(num_gen,num_genome):
    #impossible to place into the physics tick
    #search for a quicker way to save the replays
    pass
    #keyboard.press_and_release('up')
    #keyboard.press_and_release('up')
    #keyboard.press_and_release('up')
    #keyboard.press('del')
    #time.sleep(3)
    #keyboard.release('del')
    #keyboard.write(str(num_gen).zfill(3),'_'+str(num_genome).zfill(5))
    #keyboard.press_and_release('down')
    #keyboard.press_and_release('left')
    #keyboard.press_and_release('enter')
    #time.sleep(0.5)
    #keyboard.press_and_release('enter')


class MainClient(Client):
    def __init__(self):
        super(MainClient,self).__init__()

    def on_registered(self, iface: TMInterface):
        print(f'Registered to {iface.server_name}')
        #set gamespeed
        iface.set_speed(5)
        #set map start
        

    def on_deregistered(self,iface):
        print(f'deregistered to {iface.server_name}')

    def on_shutdown(self, iface):
        pass

    def on_run_step(self, iface, _time:int):
        iface.log("youpi")
        #update nn situation
        #input to nn
        #output from nn
        #execute the output
        pass

    def on_simulation_begin(self, iface):
        pass

    def on_simulation_step(self, iface, _time:int):
        pass

    def on_simulation_end(self, iface, result:int):
        pass

    def on_checkpoint_count_changed(self, iface, current:int, target:int):
        #add reward to NN
        #choose to stop the run or not
        pass

    def on_lap_count_changed(self, iface, current:int):
        #add reward to NN
        #choose to stop the run or not
        pass

    def on_custom_command(self, iface, time_from: int, time_to: int, command: str, args: list):
        """
        Called when a custom command has been executed by the user.
        Args:
            iface (TMInterface): the TMInterface object
            time_from (int): if provided by the user, the starting time of the command, otherwise -1
            time_to (int): if provided by the user, the ending time of the command, otherwise -1
            command (str): the command name being executed
            args (list): the argument list provided by the user
        """
        pass

    
    def on_client_exception(self, iface, exception: Exception):
        """
        Called when a client exception is thrown. This can happen if opening the shared file fails, or reading from
        it fails.
        Args:
            iface (TMInterface): the TMInterface object
            exception (Exception): the exception being thrown
        """
        print(f'[Client] Exception reported: {exception}')

class GenClient(Client):
    def __init__(self,L_net,max_steps):
        super(GenClient,self).__init__()
        self.respawn_steps=4*60
        self.fitness=0
        self.max_steps=max_steps+self.respawn_steps
        self.current_i=0
        self.net=L_net[0]
        self.max_i=len(L_net)
        self.current_step=0
        self.skip_frames=skip_frames
        self.L_coords=[[] for i in range(len(L_net))]
        self.L_speeds=[[] for i in range(len(L_net))]
        self.L_fit=[0 for i in range(len(L_net))]
        self.L_net=L_net
        self.checkpoint_count=0

    def on_registered(self, iface: TMInterface):
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        iface.set_speed(gamespeed)
        iface.give_up()
        

    def on_deregistered(self,iface):
        print(f'deregistered to {iface.server_name}')

    def on_shutdown(self, iface):
        pass

    def on_run_step(self, iface, _time:int):
        state=iface.get_simulation_state()
        speed=state.velocity
        self.L_speeds[self.current_i].append(speed)
        self.L_coords[self.current_i].append(state.position)
        speed=sum([abs(speed[i]) for i in range(3)])
        self.speed=speed
        if self.current_step<self.respawn_steps:
            self.current_step+=1
            if self.current_step==1:
                self.img = ImageGrab.grab()
                self.img = mod_shrink_n_measure(self.img, image_width, image_height, no_lines)
                try:
                    self.img = self.img / 255.0
                except:
                    self.img = self.img
                self.img.append(speed)
                output = np.array(self.net.activate(self.img))
                #accelerate - brake
                self.accelerate=output[1]>0
                self.brake= output[2]>0
                #steer
                steer = output[0]
                self.steer = int(steer*65536)

        else:
            #init img
            if self.current_step%self.skip_frames==0:
                self.img = ImageGrab.grab()
                #self.img.save("test.png")
                self.img = mod_shrink_n_measure(self.img, image_width, image_height, no_lines)
                #Image.fromarray(np.array(self.img)).save("test2.png")
                try:
                    self.img = self.img / 255.0
                except:
                    self.img = self.img
                self.img.append(speed)
                output = np.array(self.net.activate(self.img))
                #accelerate - brake
                self.accelerate=output[1]>0
                self.brake= output[2]>0
                #steer
                steer = output[0]
                self.steer = int(steer*65536)

            iface.set_input_state(accelerate=self.accelerate,brake=self.brake,steer=self.steer)
            #score for moving up to 0.04 per frame
            self.fitness+=self.speed/10000

            self.current_step+=1
            #update nn situation
            #input to nn
            #output from nn
            #execute the output
            if self.current_step>=self.max_steps:
                iface.log(str(self.fitness))
                self.checkpoint_count=0
                save_replay(gen,self.current_i)
                iface.give_up()
                self.L_fit[self.current_i]=self.fitness
                self.fitness=0
                self.current_i+=1

                if self.current_i<self.max_i:
                    self.net=self.L_net[self.current_i]
                    self.current_step=0
                else:
                    iface.close()
            else:
                if self.current_step>self.respawn_steps+kill_steps and self.speed<kill_speed:
                    #iface.log(str(self.speed))
                    iface.log(str(self.fitness))
                    self.checkpoint_count=0
                    save_replay(gen,self.current_i)
                    iface.give_up()
                    self.L_fit[self.current_i]=self.fitness
                    self.fitness=0
                    self.current_i+=1

                    if self.current_i<self.max_i:
                        self.net=self.L_net[self.current_i]
                        self.current_step=0
                    else:
                        iface.close()

    def on_simulation_begin(self, iface):
        pass

    def on_simulation_step(self, iface, _time:int):
        pass

    def on_simulation_end(self, iface, result:int):
        #pas call dans les run
        pass

    def on_checkpoint_count_changed(self, iface, current:int, target:int):
        self.fitness+=1000/(self.current_step/60)
        self.checkpoint_count+=1
        if self.checkpoint_count==3:#case of a finish on A01
            self.checkpoint_count=0
            self.fitness+=10000/(self.current_step/60)
            iface.log(str(self.fitness))
            iface.prevent_simulation_finish()
            save_replay(gen,self.current_i)
            iface.give_up()
            self.L_fit[self.current_i]=self.fitness
            self.fitness=0
            self.current_i+=1

            if self.current_i<self.max_i:
                self.net=self.L_net[self.current_i]
                self.current_step=0
            else:
                iface.close()
        #add reward to NN
        #choose to stop the run or not
        pass

    def on_lap_count_changed(self, iface, current:int):
        #pas call quand fin de race
        iface.log('YOUPI')
        #add reward to NN
        #choose to stop the run or not
        pass

    def on_custom_command(self, iface, time_from: int, time_to: int, command: str, args: list):
        """
        Called when a custom command has been executed by the user.
        Args:
            iface (TMInterface): the TMInterface object
            time_from (int): if provided by the user, the starting time of the command, otherwise -1
            time_to (int): if provided by the user, the ending time of the command, otherwise -1
            command (str): the command name being executed
            args (list): the argument list provided by the user
        """
        pass

    
    def on_client_exception(self, iface, exception: Exception):
        """
        Called when a client exception is thrown. This can happen if opening the shared file fails, or reading from
        it fails.
        Args:
            iface (TMInterface): the TMInterface object
            exception (Exception): the exception being thrown
        """
        print(f'[Client] Exception reported: {exception}')


def run_client_gen(client: Client, server_name: str = 'TMInterface0', buffer_size=DEFAULT_SERVER_SIZE):
    """
    Connects to a server with the specified server name and registers the client instance.
    The function closes the connection on SIGBREAK and SIGINT signals and will block
    until the client is deregistered in any way. You can set the buffer size yourself to use for
    the connection, by specifying the buffer_size parameter. Using a custom size requires
    launching TMInterface with the /serversize command line parameter: TMInterface.exe /serversize=size.
    Args:
        client (Client): the client instance to register
        server_name (str): the server name to connect to, TMInterface0 by default
        buffer_size (int): the buffer size to use, the default size is defined by tminterface.constants.DEFAULT_SERVER_SIZE
    """

    iface = TMInterface(server_name, buffer_size)

    def handler(signum, frame):
        iface.close()

    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    iface.register(client)

    while iface.running:
        time.sleep(0)#0 before
    return client.L_fit,client.L_coords,client.L_speeds

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def initial_crop(img, l, u, r, d):
    img = img.crop((l, u, img.size[0]-r, img.size[1]-d))
    return img

def mod_edge(img, w, h):
    img = initial_crop(img, 0, img.size[1]//2, 0, img.size[1]//3)
    img = ImageEnhance.Contrast(img).enhance(2).convert('L')  # .filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img)
    img = cv2.medianBlur(img, 5)
    img = (img < 50) * np.uint8(255)
    return img.reshape((h, w, 1))


def mod_shrink_n_measure(img, w, h, no_lines):
    img_np = mod_edge(img, w, h)
    return find_walls(img_np, no_lines=no_lines)


def find_walls(img_np, no_lines=10, threshold=200):
    h, w, d = img_np.shape
    dx = w//no_lines

    end_points = []

    start_points = range(dx//2, w, dx)
    for start_point in start_points:
        distance = h - 1
        while distance >= 0:
            if img_np[distance][start_point] >= threshold:  # pixel threshold
                break
            distance -= 1
        distance = h - distance - 1
        end_points.append(distance * 1.0 / h)
    
    return end_points


def run_inference(img_np, end_points):
    no_lines = len(end_points)
    h, w, d = img_np.shape
    dx = w//no_lines

    if d == 1:
        img_np = np.stack((img_np,)*3, axis=-1).reshape(h, w, 3)

    start_points = range(dx//2, w, dx)
    for start_point, end_point in zip(start_points, end_points):
        distance = end_point * h
        while distance > 0:
            i = int(h - distance)
            if i >= h:
                i = h - 1
            img_np[i][start_point][0] = 0
            img_np[i][start_point][1] = 255
            img_np[i][start_point][2] = 255
            distance -= 1
    
    return img_np


def eval_genomes(genomes, config):
    global gen
    if gen<max_gen:
        gen+=1
    L_net=[]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        L_net.append(net)
    L_fit,L_coords,L_speeds=run_client_gen(GenClient(L_net,60+10*gen),server_name) #1/6 de sec en plus par génération

    for i in range(len(L_fit)):
        genomes[i][1].fitness=L_fit[i]
    filename = 'Coords/'+str(gen).zfill(3)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_coords,outfile)
    outfile.close()
    filename = 'Speeds/'+str(gen).zfill(3)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_speeds,outfile)
    outfile.close()

def run(config_file, checkpoint=None):

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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to global no generations.
    
    winner = p.run(eval_genomes, no_generations)

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




#totest: 
#on lance le NEAT
#1 client pour initialiser la simulation
#on kill le client
#1 client par genome
#on limite le nombre de steps autorisées
#on affiche dans la console le score du genome
#on affiche dans la console les resultats de la simu

def main():
    #print(f'Connecting to {server_name}...')
    #run_client(MainClient(),server_name)
    #fit=run_client_genome(GenomeClient(None,10),server_name)
    #print(fit)
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, 'config-feedforward')
    print('Press z to begin.')
    keyboard.wait('z')

    if checkpoint == None:
        for cpt in os.listdir('.'):
            if cpt[:4] == 'neat':
                os.unlink('./'+cpt)

    run(config_path, checkpoint)
    

if __name__ == '__main__':
    main()
