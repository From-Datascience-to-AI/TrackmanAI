#the goal of this script is to accurately mesure the time taken by 
#the screenshot pipeline
from operator import index
import numpy as np
import win32gui
import win32con
import win32ui
import keyboard
from time import time
import multiprocessing
from functools import partial

l=300


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
        self.L_pix_lines2=self.get_pix_lines2(n_lines)
        self.L_indexT=[tuple(np.array(line)[:,::-1].T.tolist()) for line in self.L_pix_lines]
        self.L_indexT2=[tuple(np.array(line)[:,::-1].T.tolist()) for line in self.L_pix_lines2]
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

    def getLine2(self,x1,y1,x2,y2,l):
        return [(int(x1+i*(x2-x1)/l),int(y1+i*(y2-y1)/l)) for i in range(l+1)]

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

    def intersect2(self,indexT,im):
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

    def get_raycast2(self,im,L_indexT):
        """ Gets every (corrected) raycasts on the image
        """
        #impossible to use numpy because of the len of lines not equal
        
        #return list(map(partial(self.intersect2,im=im),L_indexT)) #to test next
        L_intersect_normed=[]
        #append=L_intersect_normed.append to test next [self.intersect2(indexT,im) for iter in L_indexT]
        #try using also return[]
        for indexT in L_indexT:
            inter=self.intersect2(indexT,im)
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

    def get_pix_lines2(self,n_lines):
        """ Gets every raycasts on the image
        """
        c,L_end_points=self.get_end_points(n_lines)
        L_pix_lines=[]
        for i in range(len(L_end_points)):
            L_pix_lines.append(self.getLine2(c[0],c[1],L_end_points[i][0],L_end_points[i][1],l))
        return L_pix_lines

    def getScreenIntersect(self):
        im=self.getScreenImg()
        L_intersect=self.get_raycast(im,self.L_pix_lines)
        return L_intersect

    def getScreenIntersect_timed(self):
        a=time()
        im=self.getScreenImg()
        b=time()-a
        a=time()
        L_intersect=self.get_raycast(im,self.L_indexT)
        c=time()-a
        return L_intersect,[b,c]

    def getScreenIntersect_timed2(self):
        a=time()
        im=self.getScreenImg()
        b=time()-a
        a=time()
        L_intersect=self.get_raycast2(im,self.L_indexT2)
        c=time()-a
        return L_intersect,[b,c]


if __name__=="__main__":
    #TODO: check identical results
    sv=ScreenViewer(20,640,480,'TrackMania United Forever (TMInterface 1.3.1)')
    print('Press z to begin.')
    keyboard.wait('z')
    L_times=[]
    L_times2=[]
    for i in range(100):
        L_intersect,L_ab=sv.getScreenIntersect_timed()
        L_intersect2,L_ab2=sv.getScreenIntersect_timed2()
        L_times.append(L_ab)
        L_times2.append(L_ab2)

    print("for current implementation")
    avg_a=sum([L_times[i][0] for i in range(len(L_times))])/len(L_times)
    avg_b=sum([L_times[i][1] for i in range(len(L_times))])/len(L_times)
    max_a=max([L_times[i][0] for i in range(len(L_times))])
    max_b=max([L_times[i][1] for i in range(len(L_times))])
    max_c=max([L_times[i][0]+L_times[i][1] for i in range(len(L_times))])
    avg_c=sum([L_times[i][0]+L_times[i][1] for i in range(len(L_times))])/len(L_times)
    print(f"average time taken for screenshot ={avg_a}")
    print(f"average time taken for screenshot processing ={avg_b}")
    print(f"average time taken for screenshot pipeline ={avg_c}")
    print(f"average framerate for screenshot pipeline ={1/avg_c}")
    print(f"minimum framerate for screenshot pipeline ={1/max_c}")
    print()
    print("for test implementation")
    avg_a2=sum([L_times2[i][0] for i in range(len(L_times2))])/len(L_times2)
    avg_b2=sum([L_times2[i][1] for i in range(len(L_times2))])/len(L_times2)
    max_a2=max([L_times2[i][0] for i in range(len(L_times2))])
    max_b2=max([L_times2[i][1] for i in range(len(L_times2))])
    max_c2=max([L_times2[i][0]+L_times2[i][1] for i in range(len(L_times2))])
    avg_c2=sum([L_times2[i][0]+L_times2[i][1] for i in range(len(L_times2))])/len(L_times2)
    print(f"average time taken for screenshot ={avg_a2}")
    print(f"average time taken for screenshot processing ={avg_b2}")
    print(f"average time taken for screenshot pipeline ={avg_c2}")
    print(f"average framerate for screenshot pipeline ={1/avg_c2}")
    print(f"minimum framerate for screenshot pipeline ={1/max_c2}")
    print()
    print("comparison")
    print(f"average time taken for screenshot gain ={avg_a-avg_a2}")
    print(f"average time taken for screenshot processing gain ={avg_b2-avg_b}")
    print(f"average time taken for screenshot pipeline gain ={avg_c-avg_c2}")
    print(f"average framerate for screenshot pipeline gain ={1/avg_c2-1/avg_c}")
    print(f"minimum framerate for screenshot pipeline gain ={1/max_c2-1/max_c}")
    print()
    print(f"average time taken for screenshot gain% ={100*avg_a/avg_a2-100}")
    print(f"average time taken for screenshot processing gain% ={100*avg_b/avg_b2-100}")
    print(f"average time taken for screenshot pipeline gain% ={100*avg_c/avg_c2-100}")
    print(f"average framerate for screenshot pipeline gain% ={100*(1/avg_c)/(1/avg_c2)-100}")
    print(f"minimum framerate for screenshot pipeline gain% ={100*(1/max_c)/(1/max_c2)-100}")