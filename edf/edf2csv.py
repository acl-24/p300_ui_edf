import numpy as np
import mne

edf = mne.io.read_raw_edf('example.edf')
data_array = edf.get_data().T 
header = ','.join(edf.ch_names)
with open('example.csv', 'w'):  
    pass
np.savetxt('example.csv', data_array, delimiter=',', header=header)
