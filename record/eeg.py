import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class EEG:
    """
    Stores EEG data and associated metadata. Handles loading EEG from a collection of `.csv` files,
    as well as some simple filtering.
    """

    X: np.ndarray  # EEG data (shape: trials, samples, channels)
    y: np.ndarray  # Ground-truth y-value for each trial (shape: trials,)
    montage: np.ndarray  # Channel names (shape: channels,)
    stimuli: np.ndarray  # Frequencies corresponding to each stimulus
    fs: float  # Sampling frequency

    @property
    def n_trials(self):
        return self.X.shape[0]

    @property
    def n_samples(self):
        return self.X.shape[1]

    @property
    def n_channels(self):
        return self.X.shape[2]

    def bandpass(self, window):
        """ Applies butterworth band-pass filter
        :param window: Tuple (low_hz, high_hz) defining the bandpass window
        :return: New EEG instance with bandpass filter applied
        """
        sos = signal.butter(5, window, btype='band', fs=self.fs, output='sos')
        X = signal.sosfiltfilt(sos, self.X, axis=1, padtype='odd')

        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

    def notch(self, freq):
        """ Applies notch filter
        :param freq: Frequency to filter out
        :return: New EEG instance with notch filter applied
        """
        b, a = signal.iirnotch(freq, 30, self.fs)
        X = signal.lfilter(b, a, self.X, axis=1)

        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

    @classmethod
    def load(cls, path, *, fs=300, epoch_start=0, epoch_length=8):
        """
        Loads EEG data from a given path, splitting it into trials according to the associated markers.

        Expects the EEG to be in `[path]/data.csv`, a csv file with headers. Timestamp data is extracted from the header
        "timestamp", which must be present. The header "TRG", if present, is ignored (this corresponds to trigger data,
        not a channel, which `EEG` does not use. The rest of the headers are expected to correspond to channels names,
        which will populate the `EEG.montage` field. The rows must be sorted by increasing timestamp.

        The markers are expected to be in `[path]/markers.csv`, which must have two columns: "timestamp" and "marker".
        Each row corresponds to a trial, with timestamps in the range [`row.timestamp+epoch_start`,
        `row.timestamp+epoch_start+epoch_length`). The trial's ground-truth y-value is `int(row.marker)`.

        The stimulation frequencies are extracted from `[path]/frequencies.txt`, which must contain a single line of
        comma-separated frequency values.

        :param path: Path to directory containing the EEG data files
        :param fs: Sampling frequency
        :param epoch_start: Offset from marker timestamp to trial start
        :param epoch_length: Size of trial epoch
        :return: Loaded EEG instance
        """
        path = Path(path)
        epoch_size = int(epoch_length * fs)

        data_table = pd.read_table(path / 'data.csv', sep=',', dtype=float, on_bad_lines='skip', engine='python')
        data_table.rename(columns=lambda s: s.strip(), inplace=True)
        markers_table = pd.read_table(path / 'markers.csv', sep=', ', engine='python')

        montage = data_table.keys()
        montage = montage[~montage.isin(['timestamp', 'TRG'])]

        try:
            class_stimuli = np.loadtxt(path / 'frequencies.txt', delimiter=',')
        except OSError:
            warnings.warn('Cannot load frequencies.txt, assuming legacy values of [12, 13, 14, 15]')
            class_stimuli = np.array([12, 13, 14, 15])

        X, y = [], []
        for _, (timestamp, marker) in markers_table.iterrows():
            data = data_table[data_table.timestamp >= timestamp]
            data = data.iloc[epoch_start:epoch_start + epoch_size]

            if data.shape[0] < epoch_size:
                print(f'Skipping marker={marker}', file=sys.stderr)
                continue

            data = data[montage]

            X.append(data)
            if isinstance(marker, str):
                y.append(np.where(np.array(['left', 'right', 'top', 'bottom']) == marker)[0][0])
                warnings.warn(f'Interpreting legacy marker type {marker} as class {y[-1]}')
            else:
                y.append(int(marker))

        return cls(np.array(X), np.array(y), montage, class_stimuli, fs)

    @classmethod
    def load_stream(cls,  path, *, fs=300):
        path = Path(path)

        data_table = pd.read_table(path / 'data.csv', sep=',', dtype=float, on_bad_lines='skip', engine='python')
        data_table.rename(columns=lambda s: s.strip(), inplace=True)
        markers_table = pd.read_table(path / 'markers.csv', sep=', ', engine='python')

        montage = data_table.keys()
        montage = montage[~montage.isin(['timestamp', 'TRG'])]

        try:
            class_stimuli = np.loadtxt(path / 'frequencies.txt', delimiter=',')
        except OSError:
            warnings.warn('Cannot load frequencies.txt, assuming legacy values of [12, 13, 14, 15]')
            class_stimuli = np.array([12, 13, 14, 15])

        X = np.array(data_table[montage])
        y = np.zeros(X.shape[0]); y[:] = np.nan

        marker_ranges = [
            (markers_table.iloc[i].marker, (markers_table.iloc[i].timestamp, markers_table.iloc[i+1].timestamp))
            for i in range(markers_table.shape[0]-1)
        ]
        marker_ranges.append((markers_table.iloc[-1].marker, (markers_table.iloc[-1].timestamp, np.inf)))

        for marker, (t_start, t_end) in marker_ranges:
            filter_ = (data_table.timestamp >= t_start) & (data_table.timestamp < t_end)
            filter_ = np.array(filter_)
            if isinstance(marker, str):
                y[filter_] = np.where(np.array(['left', 'right', 'top', 'bottom']) == marker)[0][0]
                warnings.warn(f'Interpreting legacy marker type {marker} as class {y[filter_]}')
            else:
                y[filter_] = int(marker)

        X, y = X[np.newaxis], y[np.newaxis]

        return cls(X, y, montage, class_stimuli, fs)

    def __getitem__(self, item):
        return EEG(
            X=self.X[item],
            y=self.y[item[0] if isinstance(item, tuple) else item],
            montage=self.montage,
            stimuli=self.stimuli,
            fs=self.fs,
        )


def chunkify(eeg: EEG, window_size: float, stride: float = 0.25):
    """ Splits EEG trials by a sliding window """
    X, y = [], []
    for i_trial in range(eeg.n_trials):
        for i_sample in np.arange(0, eeg.n_samples - eeg.fs * window_size, stride * eeg.fs):
            window = int(i_sample), int(i_sample + window_size * eeg.fs)
            if window[1] > eeg.n_samples:
                print('...')

            assert window[1] <= eeg.n_samples
            X.append(eeg.X[i_trial, window[0]:window[1], :])
            y.append(eeg.y[i_trial])
    X, y = np.array(X), np.array(y)
    return EEG(X=X, y=y, montage=eeg.montage, stimuli=eeg.stimuli, fs=eeg.fs)