
from encodings import normalize_encoding
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
max_time=35
no_lines = 20 #need to investigate to upscale that
filename_prefix = "models/NEAT/Checkpoints"
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
        self.L_pix_lines=Get_pix_lines(n_lines)
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
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        #w = self.r - self.l  #- self.br - self.bl
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        #h = self.b - self.t #- self.bt - self.bb
        #h=int(h*1.25)
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        #cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        cDC.BitBlt((0,0), (w, h), dcObj, (0, 38), win32con.SRCCOPY)
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
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
        im=im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]
        #return Image.fromarray(im,"RGB")
        return im



def get_end_points(n_rays):
    shape=(1279,959)
    start_point=(int(shape[0]/2),shape[1]-60) #OK
    L_teta=[ np.pi*i/(n_rays-1) for i in range(n_rays)] #OK
    ll=((start_point[0]-0)**2 + (start_point[1]-0)**2)**0.5
    lr=((start_point[0]-1279)**2 + (start_point[1]-0)**2)**0.5
    tetal=np.arccos(-(start_point[0]-0)/ll)#ok
    tetar=np.arccos(-(start_point[0]-1279)/lr)#ok
    L_end_points=[]
    for teta in L_teta:
        if teta<=tetar:
            x=1279
            if x-start_point[0]!=0:
                l=abs((x-start_point[0])/np.cos(teta))
                y=-l*np.sin(teta)+start_point[1]
            else:
                y=start_point[1]
        elif teta>=tetal:
            x=0
            if x-start_point[0]!=0:
                l=abs((x-start_point[0])/np.cos(teta))
                y=-l*np.sin(teta)+start_point[1]
            else:
                y=start_point[1]
        else:
            y=0
            if abs(y-start_point[1])>1:
                l=abs((y-start_point[1])/np.sin(teta))
                x=l*np.cos(teta)+start_point[0]
            else:
                x=start_point[0]
        L_end_points.append([x,y])
    return start_point,L_end_points

#based from https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python?fbclid=IwAR2y-tW6Qmk_1I28KQRF2WslyfmXAFhlQ3_2l0tKL8RQ7qAIj-f6QgBE-NM
def getLine(x1,y1,x2,y2): #seems good
    if x1==x2: ## Perfectly horizontal line, can be solved easily
        return [[int(x1),int(i)] for i in range(y1,y2,int(abs(y2-y1)/(y2-y1)))]
    else: ## More of a problem, ratios can be used instead
        x_inv=False
        if x1>x2: ## If the line goes "backwards", flip the positions, to go "forwards" down it.
            x=x1
            x1=x2
            x2=x
            y=y1
            y1=y2
            y2=y
            x_inv=True
        slope=(y2-y1)/(x2-x1) ## Calculate the slope of the line
        line=[]
        i=0
        while x1+i < x2: ## Keep iterating until the end of the line is reached
            i+=1
            x_end=x1+i
            y_end=y1+slope*i
            line.append([int(x_end),int(y_end)]) ## Add the next point on the line
        if x_inv:
            line.reverse()
        return line ## Finally, return the line!

def intersect(im,line):
    for i in range(len(line)):
        pix=line[i]
        shape=im.shape
        #print(shape)
        if pix[0]<shape[1] and pix[1]<shape[0]:
            color=sum(im[pix[1]][pix[0]])/len(im[pix[1]][pix[0]])
            #print(im[pix[0]][pix[1]])
            #print(color)
            if color<40:
                return i
        else:
            print(shape)
            print(pix)
    return len(line)

def Get_Raycast(im,L_pix_lines):
    L_intersect_normed=[]
    for i in range(len(L_pix_lines)):
        inter=intersect(im,L_pix_lines[i])
        L_intersect_normed.append(inter/len(L_pix_lines[i]))
    return L_intersect_normed

def Get_pix_lines(n_lines):
    c,L_end_points=get_end_points(n_lines)
    L_pix_lines=[]
    for i in range(len(L_end_points)):
        L_pix_lines.append(getLine(c[0],c[1],L_end_points[i][0],L_end_points[i][1]))
    return L_pix_lines


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
                self.im=self.sv.GetScreenImg() #try
                #self.img=Image.fromarray(self.img,"RGB")
                #self.img.save(f"{self.time}".zfill(10)+".png")
                #self.img = ImageGrab.grab()
                #Image.new(self.img).save(f"{self.time}".zfill(10)+".png")
                self.L_pix_lines=self.sv.L_pix_lines
                #self.img = ImageGrab.grab()
                self.L_raycast = Get_Raycast(self.im,self.L_pix_lines)
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
                #one screenshot every skip_frame frames
                if self.current_step%self.skip_frames==0:
                    self.im=self.sv.GetScreenImg()
                    #self.img.save("test.png") #to work on the frame processing
                    self.L_pix_lines=self.sv.L_pix_lines
                    self.L_raycast = Get_Raycast(self.im,self.L_pix_lines)
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
