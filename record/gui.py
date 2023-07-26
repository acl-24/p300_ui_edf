from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import *
import sys
import random
import pyedflib
import pandas as pd
import numpy as np
 
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 
        # Setting the title
        self.setWindowTitle("P300")

        self.data_stream = []
        self.frequency = 25 # interval of the timer and recording, in ms, which is also the block size when writing to .edf
        self.interval = 500 # interval between the visual stimulus, in ms
 
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

        # Set the selected rectangle to (potentially) change every second
        self.blinkTimer = QTimer(self, interval=self.frequency) # Set the selected rectangle to (potentially) change every second
        self.blinkTimer.timeout.connect(self.selectRectangle)
        self.blinkTimer.start()

        b1 = QPushButton('Write Stream', self)
        b1.clicked.connect(self.on_write_button_clicked)

        # Rectangles
        r1 = QRect(QPoint(self.size().width() // 5, self.size().height() // 4), QSize(self.size().width() // 5, self.size().height() // 2))
        r2 = QRect(QPoint((self.size().width() // 5) * 3, self.size().height() // 4), QSize(self.size().width() // 5, self.size().height() // 2))
       
        self.rects = [r1, r2]

        # Used if blinking the rectangles in order
        self.index = 0
       
 
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
        if (len(self.data_stream) + 1) % (self.interval / self.frequency) == 0 or len(self.data_stream) == 0:
            # Randomly select the next rectangle
            selectedIndex = self.index

            while selectedIndex == self.index:
                selectedIndex = random.randint(0, len(self.rects) - 1)
        
            self.index = selectedIndex
            
            # Select the next rectangle in the list
            ##################
            # Uncomment the line below if blinking the rectangles in order
            ##################
            # self.index = (self.index + 1) % len(self.rects)
            self.data_stream.append(selectedIndex + 1)
        else:
            self.data_stream.append(0)

    def on_write_button_clicked(self):
        num_channels = 1

        data_array = np.array(self.data_stream).astype(np.float64)
        physical_min, physical_max = np.min(data_array), np.max(data_array)

        # Create a new EDF file
        with pyedflib.EdfWriter("example.edf", n_channels=num_channels) as writer:
            signal_headers = []
            signal_labels = ['Ref']
            for i in range(num_channels):
                signal_headers.append({'label': signal_labels[i], 'dimension': "Type", 'sample_rate': self.frequency,
                                    'physical_min': physical_min, 'physical_max': physical_max,
                                    'digital_min': -32768, 'digital_max': 32767,
                                    'transducer': "", 'sample_size': len(data_array)})

            writer.setSignalHeaders(signal_headers)

            for i in range(0, len(data_array), self.frequency):
                block_end = min(i + self.frequency, len(data_array))
                block_data = data_array[i:block_end]
                writer.blockWritePhysicalSamples(block_data)

        print(len(data_array))