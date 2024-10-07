# Script from Dr Hassall, rewritten in Python with the MNE-Python library
# Compute and return the TRF for a dataset with synchronized Audio and subsequently evoked EEG

import sys
import mne
import pyxdf
import numpy as np
import matplotlib.pyplot as plt

# Load data

def main():
    if not len(sys.argv) == 2:
        print("Invalid syntax. python analyze_realworld.trf.py [filename]")
        exit(0)
    dataFile = sys.argv[1]

    ## Load data to streams, header
    streams, header = pyxdf.load_xdf(dataFile)

    eeg_stream_names = [str(ch["label"][0]) for ch in streams[0]["info"]["desc"][0]["channels"][0]["channel"]]
    print(eeg_stream_names)

    eeg = streams[0]["time_series"].T
    eeg -= eeg[-1] # Reference to 5
    eeg *= 1e-6 / 50 / 2
    print(streams[0].__repr__())
    sfreq = float(streams[0]["info"]["nominal_srate"][0])
    info = mne.create_info(eeg_stream_names, sfreq, ["eeg", "eeg", "eeg", "eeg", "eeg"])
    raw = mne.io.RawArray(eeg, info)
    raw.plot(scalings=dict(eeg=1e-6), duration=2, start=0, block=True)


if __name__ == "__main__":
    main()


# Load data
# Preprocessing
# Envelope settings
# Create Empty TRF
# Add event to EEG
# Flag bad segments of data
# Time window for TRF
# Add audio signal to data
# Generate design matrix w/ recorded data
# Perform artifact rejection on EEG
