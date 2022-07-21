from calendar import c
from turtle import title
import win32gui
import win32con
import win32ui
from threading import Thread, Lock
from PIL import ImageGrab, ImageEnhance, Image
from time import time
import keyboard
import numpy as np
import dxcam

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
        t=time()-a
        #print(im[0:3][0:3])
        return t,Image.fromarray(im,"RGB")

def GetTMShape():
    hwnd = win32gui.FindWindow(None, "TrackMania United Forever (TMInterface 1.2.0)")
    if hwnd == 0:
        hwnd = None
        return False
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    return (l,t,r,b)

if __name__=="__main__":
    sv=ScreenViewer()
    sv.GetHWND('TrackMania United Forever (TMInterface 1.2.0)')
    camera=dxcam.create()
    region=GetTMShape()
    n_img=0
    print('Press z to begin.')
    keyboard.wait('z')
    L_t=[]
    L_t2=[]
    L_t3=[]
    L_t4=[]
    for i in range(10):
        n_img+=1

        a=time()
        t,img=sv.GetScreenImg() #try
        t2=time()-a
        img.save(f"GetScreenIMG_{n_img}".zfill(10)+".png")

        a=time()
        img2=ImageGrab.grab()
        t3=time()-a
        img2.save(f"Image_grab_{n_img}".zfill(10)+".png")

        a=time()
        im=camera.grab(region=region)
        t4=time()-a
        img3=Image.fromarray(im,"RGB")
        img3.save(f"camera_grab_{n_img}".zfill(10)+".png")
        L_t.append(t)
        L_t2.append(t2)
        L_t3.append(t3)
        L_t4.append(t4)
    print(f"{sum(L_t)/len(L_t)} seconds for wingui screenshot without Image conversion")
    print(f"{sum(L_t2)/len(L_t2)} seconds for wingui screenshot with Image conversion")
    print(f"{sum(L_t3)/len(L_t3)} seconds for Imagegrab screenshot")
    print(f"{sum(L_t4)/len(L_t4)} seconds for dxcam screenshot")
    print(f"{1/(sum(L_t)/len(L_t))} screenshots per second for wingui screenshot without Image conversion")
    print(f"{1/(sum(L_t2)/len(L_t2))} screenshots per second for wingui screenshot with Image conversion")
    print(f"{1/(sum(L_t3)/len(L_t3))} screenshots per second for for Imagegrab screenshot")
    print(f"{1/(sum(L_t4)/len(L_t4))} screenshots per second for for dxcam screenshot")
    del camera
    #paralel processing dxcam:

#issue wwith h and w, got 1038 instead of 1280 and 806 instead of 960
#choice made to fi"the dimention of the window
#most of the time taken is to convert the array to image

#todo add mss screenshot
#todo add D3DShot


#todo project: use the values of pixels in high contrast to try to highlight the road or to feed it in the NN
#idea: use algorithm of movement between two successives pictures to get an Idea of the real pixel movement and use it to feed the NN
