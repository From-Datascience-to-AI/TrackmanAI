"""
Utils for TrackManAI
"""

# Imports
import os
import numpy as np
import win32gui
import win32con
import win32ui
from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE
import time
import signal
import pickle as pickle


# Classes
class ScreenViewer:
    """ Asynchronously captures screens of a window. Provides functions for accessing the captured screen.
    (from https://nicholastsmith.wordpress.com/2017/08/10/poe-ai-part-4-real-time-screen-capture-and-plumbing/?fbclid=IwAR3ZHfVY2oPr1kqhq_o4EthijXh1GLDoK2FYw3bWReRWMUEBWTB8_jhwd1Q)
    """
    def __init__(self, n_lines, w, h,wname):
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
        self.w = w
        self.h = h
        self.L_pix_lines=self.get_pix_lines(n_lines)
        self.L_indexT=[tuple(np.array(line)[:,::-1].T.tolist()) for line in self.L_pix_lines]
        self.getHWND(wname)

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

    def get_end_points(self,n_rays):
        """ Gets the end points for a given number of rays

        Parameters
        ----------
        n_rays: int (number of rays to consider)

        Output
        ----------
        start_point: (int, int) (start point for rays propagation)
        L_points : Array([int, int]) (coords of every end point)
        """
        shape=(self.w-1,self.h-1)
        start_point=(int(shape[0]/2),shape[1]-60) #OK
        L_teta=[ np.pi*i/(n_rays-1) for i in range(n_rays)] #OK
        ll=((start_point[0]-0)**2 + (start_point[1]-0)**2)**0.5
        lr=((start_point[0]-shape[0])**2 + (start_point[1]-0)**2)**0.5
        tetal=np.arccos(-(start_point[0]-0)/ll)#ok
        tetar=np.arccos(-(start_point[0]-shape[0])/lr)#ok
        L_end_points=[]
        for teta in L_teta:
            if teta<=tetar:
                x=shape[0]
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


    def getLine(self,x1,y1,x2,y2):
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

    def intersect(self,indexT,im):
        """ Gets the intersection index between the line and a different color object (i.e. a wall)

        Parameters
        ----------
        im: np.ndarray (screen image)
        line: Array([int, int])

        Output
        ----------
        i: int (index on the line that intersects a different object)
        """
        pixels=im[indexT]
        pixels=np.sum(pixels,axis=1)/len(im[0][0])
        mask=pixels<30
        mask=mask.tolist()
        mask.append(True)#end of axis 1 is True
        return 2*mask.index(True)/len(indexT[0])-1

    def get_raycast(self,im,L_indexT):
        """ Gets every (corrected) raycasts on the image
        """
        #impossible to use numpy because of the len of lines not equal
        
        #return list(map(partial(self.intersect2,im=im),L_indexT)) #to test next
        L_intersect_normed=[]
        #append=L_intersect_normed.append to test next [self.intersect2(indexT,im) for iter in L_indexT]
        #try using also return[]
        for indexT in L_indexT:
            inter=self.intersect(indexT,im)
            L_intersect_normed.append(inter)
        return L_intersect_normed

    def get_pix_lines(self,n_lines):
        """ Gets every raycasts on the image
        """
        c,L_end_points=self.get_end_points(n_lines)
        L_pix_lines=[]
        for i in range(len(L_end_points)):
            L_pix_lines.append(self.getLine(c[0],c[1],L_end_points[i][0],L_end_points[i][1]))
        return L_pix_lines

    def getScreenIntersect(self):
        im=self.getScreenImg()
        L_intersect=self.get_raycast(im,self.L_indexT)
        return L_intersect

    def getScreenIntersect_timed(self):
        a=time()
        im=self.getScreenImg()
        b=time()-a
        a=time()
        L_intersect=self.get_raycast(im,self.L_indexT)
        c=time()-a
        return L_intersect,[b,c]


class TMTrainer: #to refactor
    """ TrackManAI Trainer.
    """
    def __init__(self, model_dir, model_config, w, h, window_name, 
    n_lines, server_name, gamespeed, skip_frames, 
    kill_time, kill_speed, max_time, no_lines,
    screen_viewer, checkpoint=None):
        self.model_dir = model_dir
        self.model_config = model_config
        self.w = w
        self.h = h
        self.window_name = window_name
        self.n_lines = n_lines
        self.server_name = server_name
        self.gamespeed = gamespeed
        self.skip_frames = skip_frames
        self.kill_time = kill_time
        self.kill_speed = kill_speed
        self.max_time = max_time
        self.no_lines = no_lines
        self.screen_viewer = screen_viewer
        self.checkpoint = checkpoint
        self.filename_prefix = model_dir + "/Checkpoints/checkpoint-"
        if checkpoint == None:
            self.gen = 0
        else:
            checkpoint_infos = checkpoint.split('/')[-1].split('-')
            self.gen = int(checkpoint_infos[-1])


class Superviser:
    def __init__(self,threshold,L_maps,L_map_scores) -> None:
        self.threshold=threshold
        self.L_maps=L_maps
        self.L_map_scores=L_map_scores
        self.i=0
        self.current_map=self.L_maps[self.i]
        self.current_map_score=self.L_map_scores[self.i]

    def supervise(self,L_scores):
        n_good_scores=0
        for score in L_scores:
            if score>= self.current_map_score:
                n_good_scores+=1
        if n_good_scores/len(L_scores)>=self.threshold:
            if self.i+1<len(self.L_maps):
                self.i+=1
                self.current_map=self.L_maps[self.i]
                self.current_map_score=self.L_map_scores[self.i]
                LoadMap(self.current_map)


class MapLoader(Client):
    def __init__(self, map):
        super(MapLoader,self).__init__()
        self.map=map
        self.finished=False

    def on_registered(self, iface: TMInterface):
        """ A callback that the client has registered to a TMInterface instance.
        """
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        iface.execute_command("map "+self.map)
        iface.give_up()
        iface.close()
        self.finished=True
    

    def on_deregistered(self,iface):
        """ A callback that the client has been deregistered from a TMInterface instance. 
        This can be emitted when the game closes, the client does not respond in the timeout window, 
        or the user manually deregisters the client with the deregister command.
        """
        print(f'deregistered to {iface.server_name}')


class Logger():
    """ Logs informations about the training of a model.
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # Create folders if necessary
        if os.path.isdir(log_dir+'/Coords'):
            print("Log directory already exists, make sure you are not writting logs in an unwanted file.")
        else:
            os.mkdir(log_dir+'/Coords')
            print('Coords directory created!')

        if os.path.isdir(log_dir+'/Fitness'):
            print("Log directory already exists, make sure you are not writting logs in an unwanted file.")
        else:
            os.mkdir(log_dir+'/Fitness')
            print('Fitness directory created!')

        if os.path.isdir(log_dir+'/Inputs'):
            print("Log directory already exists, make sure you are not writting logs in an unwanted file.")
        else:
            os.mkdir(log_dir+'/Inputs')
            print('Inputs directory created!')

        if os.path.isdir(log_dir+'/Speeds'):
            print("Log directory already exists, make sure you are not writting logs in an unwanted file.")
        else:
            os.mkdir(log_dir+'/Speeds')
            print('Speeds directory created!')

    
    def log(self, log_type, L_data, i):
        """ Logs informations about a generation/epoch.
    
        Parameters
        ----------
        log_type: str (Coords - Speeds - Inputs - Fitness)
        L_data: List (array of data to log)
        i: int (index of generation/epoch)

        Output
        ----------
        None
        """
        if log_type in ['Coords', 'Speeds', 'Fitness']:
            filename = self.log_dir + '/' + log_type + '/' + str(i).zfill(5) + '.pickle'
            outfile = open(filename, 'wb')
            pickle.dump(L_data, outfile)
            outfile.close()
        elif log_type == 'Inputs':
            # Have to refine data first
            for index in range(len(L_data)):
                filename = self.log_dir + '/' + log_type + '/' + str(i).zfill(5) +'_'+ str(index).zfill(3)
                l_inputs = L_data[i]
                outfile = open(filename, 'a')
                for jndex in range(len(l_inputs)):
                    inputs = l_inputs[jndex]
                    time = inputs[0]
                    accelerate=inputs[1]
                    brake=inputs[2]
                    steer=inputs[3]
                    if accelerate:
                        outfile.write(str(time)+'press up\n')
                    if brake:
                        outfile.write(str(time)+'press down\n')
                    outfile.write(str(time)+'steer '+str(steer).zfill(5)+'\n')
                outfile.close()
        else:
            raise Exception(f'Log type name exception : log_type unknown ({log_type})')


# Functions
def LoadMap(map):
    client=MapLoader(map)
    buffer_size=DEFAULT_SERVER_SIZE
    server_name="TMInterface0"
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
    time.sleep(5)