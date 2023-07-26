import sys
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from threading import Lock
from typing import List

import numpy as np

from .app import App
from .eeg import EEG


class InputListener:
    """
    Base class for anything that receives data/marker streams from `InputDistributor`.
    """
    def ingest_data(self, timestamp: float, eeg: EEG):
        """ Receive EEG data corresponding to a single timestamp.

        :param timestamp: timestamp of input EEG data.
        :param eeg: EEG instance with n_trials=1, n_samples=1
        """
        pass

    def ingest_marker(self, timestamp: float, marker: str):
        """ Ingests a single marker. """
        pass


class FileRecorder(InputListener):
    """
    Receives EEG data from an InputDistributor and saves it to disk in a format that can be loaded with `EEG.load()`.

    The intended use case involves three threads: the input-distributor thread, on which `InputDistributor` collects
    and distributes its EEG input; the file-recorder thread, on which `FileRecorder.loop()` runs; and the main thread,
    which orchestrates the other two and eventually calls `FileRecorder.kill()` and `InputDistributor.kill()`.
    """
    def __init__(self, path: Path, channel_names: List):
        """ Create a FileRecorder instance.

        :param path: directory to save the EEG data to.
        :param channel_names: names corresponding to channels given by the InputDistributor.
        """
        self.path = path
        self.data_file = open(self.path / 'data.csv', 'w')
        self.markers_file = open(self.path / 'markers.csv', 'w')
        self.data_file.write(f'timestamp, {", ".join(channel_names)}\n')
        self.markers_file.write('timestamp, marker\n')
        self.data_buffer = Queue()
        self.markers_buffer = Queue()
        self.done = False

    def kill(self):
        """ Terminate `loop()` when buffers  """
        self.done = True

    def ingest_data(self, timestamp, eeg):
        """ Ingests EEG from the InputDistributor. """
        if not self.done:
            self.data_buffer.put((timestamp, eeg))

    def ingest_marker(self, timestamp, marker):
        """ Ingests markers from the InputDistributor. """
        if not self.done:
            self.markers_buffer.put((timestamp, marker))

    def loop(self):
        """ Loop until killed, writing ingested data and markers to disk.
        Data is written in batches of 100 to reduce I/O usage.
        """
        while not self.data_buffer.empty() or not self.markers_buffer.empty() or not self.done:
            if self.data_buffer.qsize() > 100 or (not self.data_buffer.empty() and self.done):
                try:
                    while True:
                        timestamp, eeg = self.data_buffer.get_nowait()
                        self.data_file.write(f'{timestamp}, {", ".join(str(d) for d in eeg.X.flatten())}\n')
                except Empty:
                    pass
                self.data_file.flush()

            if self.markers_buffer.qsize() > 0:
                try:
                    while True:
                        timestamp, marker = self.markers_buffer.get_nowait()
                        self.markers_file.write(f'{timestamp}, {marker}\n')
                except Empty:
                    pass
                self.markers_file.flush()

            time.sleep(0.1)


class DisconnectError(Exception):
    pass


class InputDistributor:
    """
    Receives SSVEP input from a DSIInput instance and distributes it to various listeners.

    The intended use case involves two threads: the input-distributor thread where `InputDistributor.loop()` is called,
    and the main thread, which calls orchestrates the input-distributor thread, and eventually kills it by calling
    `InputDistributor.kill()`.
    """
    def __init__(self, app: App, listeners: List[InputListener], *, disconnect_timeout=1):
        """

        :param app: App to get DSIInput instance from
        :param listeners: List of listeners to give new EEG samples to through their `ingest_data()` and
                          `ingest_marker()` methods.
        :param disconnect_timeout: Print warning if the time between received batches is larger than this. Set to None
                                   to disable.
        """
        self.app = app
        self.listeners = listeners
        self.disconnect_timeout = disconnect_timeout
        self.done = False

    def kill(self):
        """ Mark the InputDistributor as killed. """
        self.done = True

    def wait_for_connection(self):
        """ Wait until the headset starts producing data """
        while not self.done:
            for _ in self.app.dsi_input.pop_all_data():
                return
            time.sleep(0.1)

    def loop(self):
        """
        Repeatedly pops data and markers off of the DSIInput queues and pushes them to all listeners.
        Stops only when the InputDistributor is marked as killed (i.e. when `kill()` is called in a separate thread).
        """
        self.wait_for_connection()
        channel_names = np.array(self.app.dsi_input.get_channel_names())

        last_data_time = time.time()

        while not self.done:

            for timestamp, data in self.app.dsi_input.pop_all_data():
                last_data_time = time.time()
                eeg = EEG(
                    np.array(data)[np.newaxis, np.newaxis],
                    None,
                    channel_names,
                    self.app.freq,
                    self.app.dsi_input.get_sampling_rate()
                )
                for l in self.listeners:
                    l.ingest_data(timestamp, eeg)

            for timestamp, marker in self.app.dsi_input.pop_all_markers():
                last_data_time = time.time()
                for l in self.listeners:
                    l.ingest_marker(timestamp, marker)

            if self.disconnect_timeout is not None and time.time() - last_data_time > self.disconnect_timeout:
                print('Headset connection timed out')
                # raise DisconnectError()

            time.sleep(0.1)