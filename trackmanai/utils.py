"""
Utils for TrackManAI
"""

# Imports
import numpy as np
import win32gui
import win32con
import win32ui
from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE
import signal
import time


# Classes
class ScreenViewer:
    """ Asynchronously captures screens of a window. Provides functions for accessing the captured screen.
    (from https://nicholastsmith.wordpress.com/2017/08/10/poe-ai-part-4-real-time-screen-capture-and-plumbing/?fbclid=IwAR3ZHfVY2oPr1kqhq_o4EthijXh1GLDoK2FYw3bWReRWMUEBWTB8_jhwd1Q)
    """
    def __init__(self, n_lines, w, h):
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
        self.L_pix_lines=get_pix_lines(n_lines)
        self.w = w
        self.h = h
    

    def getHWND(self, wname):
        """ Gets handle of window to view.
        Also updates class parameters for the screen window on success.
    
        Parameters
        ----------
        wname: str (Title of window to find)

        Output
        ----------
        out : bool (True on success; False on failure)
        """
        self.hwnd = win32gui.FindWindow(None, wname)
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
         

    def getScreenImg(self):
        """ Gets the screen of the window referenced by self.hwnd
    
        Parameters
        ----------
        None

        Output
        ----------
        im : np.ndarray (screen image)
        """
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
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        #cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        cDC.BitBlt((0,0), (self.w, self.h), dcObj, (0, 38), win32con.SRCCOPY)
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
                self.im=self.sv.getScreenImg() #try
                #self.img=Image.fromarray(self.img,"RGB")
                #self.img.save(f"{self.time}".zfill(10)+".png")
                #self.img = ImageGrab.grab()
                #Image.new(self.img).save(f"{self.time}".zfill(10)+".png")
                self.L_pix_lines=self.sv.L_pix_lines
                #self.img = ImageGrab.grab()
                self.L_raycast = get_raycast(self.im,self.L_pix_lines)
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
                    self.im=self.sv.getScreenImg()
                    #self.img.save("test.png") #to work on the frame processing
                    self.L_pix_lines=self.sv.L_pix_lines
                    self.L_raycast = get_raycast(self.im,self.L_pix_lines)
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


# Functions
def get_end_points(n_rays):
    """ Gets the end points for a given number of rays

    Parameters
    ----------
    n_rays: int (number of rays to consider)

    Output
    ----------
    start_point: (int, int) (start point for rays propagation)
    L_points : Array([int, int]) (coords of every end point)
    """
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


def getLine(x1,y1,x2,y2):
    """ Compute a straight line from (x1, y1) to (x2, y2)
    (based on https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python?fbclid=IwAR2y-tW6Qmk_1I28KQRF2WslyfmXAFhlQ3_2l0tKL8RQ7qAIj-f6QgBE-NM)

    Parameters
    ----------
    x1: int
    y1: int
    x2: int
    y2: int

    Output
    ----------
    line: Array([int, int]) (each point from (x1, y1) to (x2, y2))
    """
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
    """ Gets the intersection index between the line and a different color object (i.e. a wall)

    Parameters
    ----------
    im: np.ndarray (screen image)
    line: Array([int, int])

    Output
    ----------
    i: int (index on the line that intersects a different object)
    """
    for i in range(len(line)):
        pix=line[i]
        shape=im.shape
        #print(shape)
        if pix[0]<shape[1] and pix[1]<shape[0]:
            color=sum(im[pix[1]][pix[0]])/len(im[pix[1]][pix[0]])
            #print(im[pix[0]][pix[1]])
            #print(color)
            if color<30:
                return i
        else:
            print(shape)
            print(pix)
    return len(line)


def get_raycast(im,L_pix_lines):
    """ Gets every (corrected) raycasts on the image
    """
    L_intersect_normed=[]
    for i in range(len(L_pix_lines)):
        inter=intersect(im,L_pix_lines[i])
        L_intersect_normed.append(2*inter/len(L_pix_lines[i])-1)
    return L_intersect_normed


def get_pix_lines(n_lines):
    """ Gets every raycasts on the image
    """
    c,L_end_points=get_end_points(n_lines)
    L_pix_lines=[]
    for i in range(len(L_end_points)):
        L_pix_lines.append(getLine(c[0],c[1],L_end_points[i][0],L_end_points[i][1]))
    return L_pix_lines


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