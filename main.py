import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import *
import sys

from record.app import Window
from record.input_distributor import FileRecorder, InputDistributor


def main(app: Window, *, path: Path):
    app.connect_to_headset()

    file_logger = FileRecorder(path, app.dsi_input.get_channel_names())
    input_distributor = InputDistributor(app, listeners=[file_logger])
    input_distributor.wait_for_connection()

    Thread(target=lambda: input_distributor.loop()).start()
    Thread(target=lambda: file_logger.loop()).start()


if __name__ == '__main__':
    config = json.load(open('config.json'))

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path(config['dataset_path']) / timestamp
    path.mkdir(parents=True, exist_ok=False)

    # create pyqt5 app
    App = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(App.exec())