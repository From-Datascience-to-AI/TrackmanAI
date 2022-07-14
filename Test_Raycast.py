import win32gui
import win32con
import win32ui
from threading import Thread, Lock
from PIL import Image
from time import time
import keyboard
import numpy as np
import matplotlib.pyplot as plt

#goal of this script: testing the screenshot part and mesuring the time taken

w=1280
h=960

class ScreenViewer:
 
    def __init__(self):
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        self.bl, self.bt, self.br, self.bb = 12, 31, 12, 20
        #self.bl, self.bt, self.br, self.bb = 0, 0, 0, 0
 
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
        a=time()
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
        dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
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
        print(time()-a)
        print(im[0:3][0:3])
        return Image.fromarray(im,"RGB")

def draw_rays(n_rays,img): #OK
    img=img.convert('L')
    shape=img.size
    start_point=(int(shape[0]/2),shape[1]-10) #OK
    plt.plot(start_point[0],start_point[1],color='r',marker='+')
    L_teta=[ np.pi*i/(n_rays-1) for i in range(n_rays)] #OK
    ll=((start_point[0]-0)**2 + (start_point[1]-0)**2)**0.5
    lr=((start_point[0]-1260)**2 + (start_point[1]-0)**2)**0.5
    tetal=np.arccos(-(start_point[0]-0)/ll)#ok
    tetar=np.arccos(-(start_point[0]-1260)/lr)#ok
    print(f"tetal={tetal}")
    print(f"tetar={tetar}")
    L_end_points=[]
    for teta in L_teta:
        if teta<=tetar:
            x=1260
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
    plt.imshow(img)
    for i in range(len(L_end_points)):
        x=L_end_points[i][0]
        y=L_end_points[i][1]
        teta=L_teta[i]
        print(f"x={x}")
        print(f"y={y}")
        #plt.axline(start_point,slope=np.tan(teta),color='r')
        plt.plot([start_point[0],x],[start_point[1],y],color='b')
    plt.show()

#from https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python?fbclid=IwAR2y-tW6Qmk_1I28KQRF2WslyfmXAFhlQ3_2l0tKL8RQ7qAIj-f6QgBE-NM
def getLine(x1,y1,x2,y2):
    if x1==x2: ## Perfectly horizontal line, can be solved easily
        return [(x1,i) for i in range(y1,y2,int(abs(y2-y1)/(y2-y1)))]
    else: ## More of a problem, ratios can be used instead
        if x1>x2: ## If the line goes "backwards", flip the positions, to go "forwards" down it.
            x=x1
            x1=x2
            x2=x
            y=y1
            y1=y2
            y2=y
        slope=(y2-y1)/(x2-x1) ## Calculate the slope of the line
        line=[]
        i=0
        while x1+i < x2: ## Keep iterating until the end of the line is reached
            i+=1
            line.append((x1+i,y1+slope*i)) ## Add the next point on the line
        return line ## Finally, return the line!

#def intersect(img,line):
#    for pix in line:
#        if pix of img is dark:
#            return pix id
#    return len(line)


if __name__=="__main__":

    sv=ScreenViewer()
    sv.GetHWND('TrackMania United Forever (TMInterface 1.1.0)')
    n_img=0
    print('Press z to begin.')
    keyboard.wait('z')
    for i in range(1):
        n_img+=1
        a=time()
        img=sv.GetScreenImg() #try
        t=time()-a
        img.save(f"{n_img}".zfill(10)+".png")
        draw_rays(10,img)
        print(t)

#issue wwith h and w, got 1038 instead of 1280 and 806 instead of 960
#choice made to fi"the dimention of the window
#most of the time taken is to convert the array to image