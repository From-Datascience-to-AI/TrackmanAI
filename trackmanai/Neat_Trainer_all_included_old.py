
from tminterface.client import Client
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
import pickle as pickle


def save_replay(num_gen,num,time_wait,init,init_gen):
    #impossible to place into the physics tick
    #search for a quicker way to save the replays
    keyboard.press_and_release('s')
    time.sleep(time_wait)
    keyboard.press('shift')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.release('shift')
    time.sleep(time_wait)
    if not init:
        if init_gen:
            for i in range(13):
                keyboard.press_and_release(14)
                time.sleep(time_wait)
            keyboard.write(str(num_gen).zfill(5)+'_'+str(num).zfill(7))
        else:
            for i in range(7):
                keyboard.press_and_release(14)
                time.sleep(time_wait)
            keyboard.write(str(num).zfill(7))
    else:
        time.sleep(time_wait)
        keyboard.write(str(num_gen).zfill(5)+'_'+str(num).zfill(7))
    time.sleep(time_wait)
    keyboard.press_and_release('Enter')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.press_and_release('tab')
    time.sleep(time_wait)
    keyboard.press_and_release('Enter')
    time.sleep(time_wait)
    keyboard.press_and_release('Enter')
    time.sleep(time_wait)
    keyboard.press_and_release('Enter')#in case of overwritting a replay
    time.sleep(time_wait)
    keyboard.press_and_release('Enter')
    time.sleep(time_wait)


KEY_ENTER = 0x1C

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
checkpoint = "neat-checkpoint-1"
gen=1 #current gen
max_gen=200
server_name=f'TMInterface{sys.argv[1]}' if len(sys.argv)>1 else 'TMInterface0'
init=True

class GenClient(Client):
    def __init__(self,L_net,max_steps):
        super(GenClient,self).__init__()
        self.respawn_steps=0 #to remove as time>0 is used
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
        self.to_save=False
        self.time=-9999
        self.ready_to_go=False
        self.ready_max_steps=60
        self.ready_current_steps=0

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
        self.time=_time
        if self.time<=0:
            self.accelerate=False
            self.brake=False
            self.steer=0
        if self.to_save:
            self.ready_current_steps+=1
            if self.ready_current_steps>self.ready_max_steps:
                self.ready_current_steps=0
                self.ready_to_go=True
        if self.time>=0:
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
                    #iface.log(str(self.fitness))
                    self.checkpoint_count=0
                    
                    self.L_fit[self.current_i]=self.fitness
                    self.fitness=0
                    self.current_i+=1

                    if self.current_i<self.max_i:
                        self.net=self.L_net[self.current_i]
                        self.current_step=0
                        self.to_save=True
                    else:
                        self.to_save=True
                        iface.close()
                else:
                    if self.current_step>self.respawn_steps+kill_steps and self.speed<kill_speed:
                        #iface.log(str(self.speed))
                        #iface.log(str(self.fitness))
                        self.checkpoint_count=0
                        
                        self.L_fit[self.current_i]=self.fitness
                        self.fitness=0
                        self.current_i+=1

                        if self.current_i<self.max_i:
                            self.net=self.L_net[self.current_i]
                            self.current_step=0
                            self.to_save=True
                        else:
                            self.to_save=True
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
        if current==target:#case of a finish
            self.checkpoint_count=0
            self.fitness+=10000/(self.current_step/60)
            #iface.log(str(self.fitness))
            iface.prevent_simulation_finish()
            
            self.L_fit[self.current_i]=self.fitness
            self.fitness=0
            self.current_i+=1

            if self.current_i<self.max_i:
                self.net=self.L_net[self.current_i]
                self.current_step=0
                self.to_save=True
            else:
                self.to_save=True
                iface.close()
                
        #add reward to NN
        #choose to stop the run or not
        pass

    def on_lap_count_changed(self, iface, current:int):
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
        #iface.register(self)#try


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
    global gen, init
    iface = TMInterface(server_name, buffer_size)

    def handler(signum, frame):
        iface.close()

    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    reg=iface.register(client)
    time.sleep(1)
    iface.execute_command("clear")
    iface.give_up()
    init_gen=True
    client.current_step=0
    for n_genome in range(100):
        print(n_genome)
        time.sleep(0.1)#seems to solve the issue
        while not client.to_save:
            #reg=iface.register(client)
            time.sleep(0)#The error comes from here => during physics step
        print("youpi0")
        save_replay(gen,n_genome,0.2,init,init_gen)
        init=False
        init_gen=False
        if n_genome<99:
            print("youpi1")
            client.to_save=False
            while client.time>0: #issues here, time does not update
                keyboard.press_and_release('enter')
                time.sleep(0.2)
            client.current_step=0
            while not client.ready_to_go:
                time.sleep(0)
            time.sleep(0.1)
            client.ready_to_go=False
            client.time=-9999
            #iface.clear_event_buffer()
            time.sleep(0.1)
            iface.give_up()
            print("youpi2")
    iface.close()

    return client.L_fit,client.L_coords,client.L_speeds


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
    gen+=1
    if gen>max_gen:
        genb=max_gen
    else:
        genb=gen
    L_net=[]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        L_net.append(net)
    L_fit,L_coords,L_speeds=run_client_gen(GenClient(L_net,180+20*genb),server_name) #1/6 de sec en plus par génération

    for i in range(len(L_fit)):
        genomes[i][1].fitness=L_fit[i]
    filename = 'NEAT/Coords/'+str(gen).zfill(3)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_coords,outfile)
    outfile.close()
    filename = 'NEAT/Speeds/'+str(gen).zfill(3)+'.pickle'
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

    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, 'NEAT/config-feedforward')
    print('Press z to begin.')
    keyboard.wait('z')

    if checkpoint == None:
        for cpt in os.listdir('.'):
            if cpt[:4] == 'neat':
                os.unlink('./'+cpt)

    run(config_path, checkpoint)
    

if __name__ == '__main__':
    main()
