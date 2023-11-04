from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow
import sys
from utils import ScreenViewer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui
import time

class ScreenThread(QtCore.QThread):#uses signal to communicate with the app
    screen_taken=QtCore.pyqtSignal(object)

    def __init__(self,parent):
        QtCore.QThread.__init__(self,parent)
        self.sv=ScreenViewer(20, 640, 480, 300, "TrackMania United Forever (TMInterface 1.3.1)")

    def run(self):
        while True:
            pts,im=self.sv.getScreenIntersect_forplot()
            self.screen_taken.emit([pts,im])
            time.sleep(0.1)



class Visualizer(QMainWindow):
    def __init__(self):
        super(Visualizer,self).__init__()

        self.sv=ScreenViewer(20, 640, 480, 300, "TrackMania United Forever (TMInterface 1.3.1)")
        img=self.sv.getScreenImg()

        self.setWindowTitle("Visualizer")
        self.setGeometry(100, 100, 640, 480)
        
        self.widget=QWidget()
        self.setCentralWidget(self.widget)

        self.layout1=QVBoxLayout()
        self.fig=Figure(figsize=(640,480))
        self.figcanvas=FigureCanvasQTAgg(self.fig)
        self.ax=self.fig.add_subplot(111)
        self.fig.subplots_adjust(0,0,1,1,0,0)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.layout1.addWidget(self.figcanvas)
        self.widget.setLayout(self.layout1)
        self.ax.imshow(img)
        #self.figcavas.clear()
        #self.figcanvas.draw()
        self.show()
        self.start_screens()

    def start_screens(self):
        screener=ScreenThread(parent=self)
        screener.screen_taken.connect(self.on_screen_ready)
        screener.start()

    def on_screen_ready(self,ptsim):
        pts=ptsim[0]
        im=ptsim[1]
        self.ax.clear()
        self.ax.set_xlim(0,640)
        self.ax.set_ylim(480,0)
        for pt in pts:
            a=pt[0]
            b=pt[1]
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color='green',marker='+')
        self.ax.imshow(im)
        self.figcanvas.draw()
        self.figcanvas.flush_events()
        
        
        



app=QApplication([])
V=Visualizer()
sys.exit(app.exec())
"""
window = QWidget()
window.setWindowTitle("Visualizer")

window.setGeometry(100, 100, 280, 80)

layout1 = QVBoxLayout()
label_map=QLabel("MAP")
label_genome=QLabel("GENOME")

layout2 = QHBoxLayout()



layout1.addWidget(label_map)
layout1.addWidget(label_genome)
window.setLayout(layout1)

window.show()

sys.exit(app.exec())
"""