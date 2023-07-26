import time
import threading
import numpy as np

from .dsi_input import DSIInput

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import *
import sys
import random
import pyedflib
import pandas as pd
import numpy as np
 
class Window(QMainWindow):
    def __init__(self, *, freq=(12, 13, 14, 15),
                 headset=None, headset_port='COM19',):
        
        super().__init__()
 
        # Setting the title
        self.setWindowTitle("P300")

        self.interval = 500 # interval between the visual stimulus, in ms
        self.freq = np.array(freq)
        self.done = False
        self.dsi_input = headset if headset is not None else DSIInput()
        self.headset_port = headset_port
 
        # Setting geometry to main window
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowState(Qt.WindowMaximized)
 
        # Creating an image object for the canvas
        screen_size = self.size()
        canvas_size = QSize(screen_size.width(),  screen_size.height())
        self.image = QImage(canvas_size, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.headset_thread = QThread(target=self.headset_thread)
        self.experiment_thread = QThread(target=self.experiment_thread)

        try:
            self.headset_thread.start()
            self.experiment_thread.start()
        except Exception as ex:
            print("Error when starting headset and marker thread!")

        # Set the selected rectangle to (potentially) change every second
        self.blinkTimer = QTimer(self, interval=self.interval) # Set the selected rectangle to (potentially) change every second
        self.blinkTimer.timeout.connect(self.selectRectangle)
        self.blinkTimer.start()

        # Rectangles
        r1 = QRect(QPoint(self.size().width() // 5, self.size().height() // 4), QSize(self.size().width() // 5, self.size().height() // 2))
        r2 = QRect(QPoint((self.size().width() // 5) * 3, self.size().height() // 4), QSize(self.size().width() // 5, self.size().height() // 2))
       
        self.rects = [r1, r2]

        # Used if blinking the rectangles in order
        self.index = 0
        self.mutex = threading.Lock()
       
    def kill(self):
        self.headset_thread.join()
        self.experiment_thread.join()

        self.close()

    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        self.image.fill(Qt.black)
        canvasPainter.end()

        # Adjust the height of the rectangles for if the screen size adjusts
        h = self.image.size().height()
        w = self.image.size().width()

        r1 = QRect(QPoint(w // 5, h // 4), QSize(w // 10, h // 4))
        r2 = QRect(QPoint((w // 5) * 3, h // 4), QSize(w // 10, h // 4))

        self.rects = [r1, r2]

        # Draw the Rectangles on the canvas in black
        canvasPainter = QPainter(self.image)

        canvasPainter.setPen(QPen(QColor(191,191,191), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        canvasPainter.setBrush(QBrush(QColor(191,191,191), Qt.SolidPattern))

        for rect in self.rects:
            canvasPainter.drawRect(rect)

        # Draw the selected rectangle on the canvas in red
        canvasPainter.setPen(QPen(QColor(255,255,255), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        canvasPainter.setBrush(QBrush(QColor(255,255,255), Qt.SolidPattern))

        canvasPainter.drawRect(self.rects[self.index])

        canvasPainter.end()

        self.update()

    def selectRectangle(self):
        # Randomly select the next rectangle
        selectedIndex = self.index

        while selectedIndex == self.index:
            selectedIndex = random.randint(0, len(self.rects) - 1)
    
    
        self.mutex.acquire()
        self.index = selectedIndex
        self.mutex.release()
        
        # Select the next rectangle in the list
        ##################
        # Uncomment the line below if blinking the rectangles in order
        ##################
        # self.index = (self.index + 1) % len(self.rects)

    def headset_thread(self):
        try:
            while not self.dsi_input.is_attached() and not self.done:
                time.sleep(0.1)
            if self.dsi_input.is_attached():
                self.dsi_input.loop()
                time.sleep(0.1)
        except Exception as ex:
            print('Exception in headset_thread:', ex)
        finally:
            self.kill()
            print('headset_thread finished!')

    def experiment_thread(self):
        try:
            while not self.done:
                if self.dsi_input.is_attached():
                    self.mutex.acquire()
                    if self.index != 0:                       
                        timestamp = self.dsi_input.get_latest_timestamp()
                        self.dsi_input.push_marker(timestamp, self.index)
                        self.index = 0
                    self.mutex.release()
        except Exception as ex:
            print('Exception in experiment_thread:', ex)
        finally:
            self.kill()
            print('experiment_thread finished!')

    def set_text(self, str):
        print(f'Text: "{str}"')

    def connect_to_headset(self, retry=True):
        if self.dsi_input.is_attached():
            return

        self.set_text('Connecting to headset')
        while not self.done:
            try:
                self.dsi_input.attach(self.headset_port)
                break
            except Exception as ex:
                print('Exception in headset attach:', ex)
                if not retry:
                    break
                time.sleep(1)
        self.set_text('')
