
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

#os.environ["PATH"] = r"d:D:\ProgramData\Anaconda3\envs\trackmanAIenv\Lib\site-packagespywin32_system32;" + os.environ["PATH"]

import win32gui
import win32con
import win32ui
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

# image dimensions
image_width = 100
image_height = 100

# hyperparams
#threshold = 0.5
no_generations = 1000 #20 for straight with 8sec of max
#max_fitness = 100.0
gamespeed=1
skip_frames=5
kill_time= 3
kill_speed = 18
max_time=35
no_lines = 20 #need to investigate to upscale that
filename_prefix = "models/NEAT/mlruns/training_steer_gas/"
checkpoint = None # filename_prefix + "neat-checkpoint-0"
gen=319 #current gen
server_name=f'TMInterface{sys.argv[1]}' if len(sys.argv)>1 else 'TMInterface0'



#from https://nicholastsmith.wordpress.com/2017/08/10/poe-ai-part-4-real-time-screen-capture-and-plumbing/?fbclid=IwAR3ZHfVY2oPr1kqhq_o4EthijXh1GLDoK2FYw3bWReRWMUEBWTB8_jhwd1Q

#Asynchronously captures screens of a window. Provides functions for accessing
#the captured screen.

class ScreenViewer:
 
    def __init__(self):
        #self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        #self.bl, self.bt, self.br, self.bb = 12, 31, 12, 20
        self.bl, self.bt, self.br, self.bb = 0, 0, 0, 0
 
    #Gets handle of window to view
    #wname:         Title of window to find
    #Return:        True on success; False on failure
    def GetHWND(self, wname):
        self.hwnd = win32gui.FindWindow(None, wname)
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
         
         
    #Gets the screen of the window referenced by self.hwnd
    def GetScreenImg(self):
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        w = self.r - self.l - self.br - self.bl
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        h = self.b - self.t - self.bt - self.bb
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        bmInfo = dataBitMap.GetInfo()
        im = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype = np.uint8)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #Bitmap has 4 channels like: BGRA. Discard Alpha and flip order to RGB
        #For 800x600 images:
        #Remove 12 pixels from bottom + border
        #Remove 4 pixels from left and right + border
        return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]

class GenClient(Client):
    def __init__(self,L_net,max_time,kill_time):
        super(GenClient,self).__init__()
        self.init_step=True
        self.fitness=0
        self.max_time=max_time*1000
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
        self.sv=ScreenViewer()
        self.sv.GetHWND('TrackMania United Forever (TMInterface 1.2.0)')

    def on_registered(self, iface: TMInterface):
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        if gamespeed!=1.0:
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
            self.yaw=0
            self.pitch=0
            self.roll=0
        if self.time>=0:
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
                self.init_step=False
                self.current_step+=1
                #self.img=self.sv.GetScreenImg() #try
                #self.img=Image.fromarray(self.img,"RGB")
                #self.img.save(f"{self.time}".zfill(10)+".png")
                self.img = ImageGrab.grab()
                #Image.new(self.img).save(f"{self.time}".zfill(10)+".png")
                
                #self.img = ImageGrab.grab()
                self.img = mod_shrink_n_measure(self.img, image_width, image_height, no_lines)
                try:
                    self.img = self.img / 255.0
                except:
                    self.img = self.img
                self.img.append(speed)
                self.img.append(self.yaw)
                self.img.append(self.pitch)
                self.img.append(self.roll)
                #iface.log(str(self.img[:-4]))
                output = np.array(self.net.activate(self.img))
                #inputs
                self.accelerate=output[1]>0
                self.brake= output[2]>0
                steer = output[0]
                self.steer = int(steer*65536)

            else:
                #one screenshot every skip_frame frames
                if self.current_step%self.skip_frames==0:
                    self.img = ImageGrab.grab()
                    #self.img.save("test.png") #to work on the frame processing
                    self.img = mod_shrink_n_measure(self.img, image_width, image_height, no_lines)
                    #Image.fromarray(np.array(self.img)).save("test2.png")
                    try:
                        self.img = self.img / 255.0
                    except:
                        self.img = self.img
                    self.img.append(speed)
                    self.img.append(self.yaw)
                    self.img.append(self.pitch)
                    self.img.append(self.roll)
                    output = np.array(self.net.activate(self.img))
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
                    if self.time>self.kill_time and self.speed<kill_speed:
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
        pass

    def on_simulation_step(self, iface, _time:int):
        pass

    def on_simulation_end(self, iface, result:int):
        #pas call dans les run
        pass

    def on_checkpoint_count_changed(self, iface, current:int, target:int):
        self.fitness+=10000/(self.time/1000)
        if current==target:#case of a finish
            self.fitness+=100000/(self.time/10000)
            #iface.log(str(self.fitness))
            iface.prevent_simulation_finish()
            
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
    iface = TMInterface(server_name, buffer_size)

    def handler(signum, frame):
        iface.close()

    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    iface.register(client)
    while not client.finished:
        time.sleep(0)
    iface.close()

    return client.L_fit,client.L_coords,client.L_speeds,client.L_inputs


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
    #print(end_points)
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
    global gen,kill_time,max_time
    gen+=1
    L_net=[]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        L_net.append(net)
    max_time2=min(max_time,1+0.5*gen)
    L_fit,L_coords,L_speeds,L_inputs=run_client_gen(GenClient(L_net,max_time2,kill_time),server_name) #1/6 de sec en plus par génération

    for i in range(len(L_fit)):
        genomes[i][1].fitness=L_fit[i]

    filename = 'NEAT/Coords/'+str(gen).zfill(5)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_coords,outfile)
    outfile.close()
    filename = 'NEAT/Speeds/'+str(gen).zfill(5)+'.pickle'
    outfile = open(filename,'wb')
    pickle.dump(L_speeds,outfile)
    outfile.close()

    for i in range(len(L_inputs)):
        filename="NEAT/Inputs/"+str(gen).zfill(5)+'_'+str(i).zfill(3)
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
