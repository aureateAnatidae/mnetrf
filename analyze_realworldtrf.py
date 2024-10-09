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

    eegStreamNames = [str(ch["label"][0]) for ch in streams[0]["info"]["desc"][0]["channels"][0]["channel"]]
    audioStreamName = [streams[1]["info"]["name"][0]]

    eeg = streams[0]["time_series"].T
    eeg *= 1e-6 / 50 / 2
    eegSfreq = float(streams[0]["info"]["nominal_srate"][0])

    eegInfo = mne.create_info(eegStreamNames, eegSfreq, ["eeg", "eeg", "eeg", "eeg", "eeg"])
    eegRaw = mne.io.RawArray(eeg, eegInfo)
    eegRaw = eegRaw.drop_channels("Right AUX") # We are not using this "electrode"

    audio = streams[1]["time_series"].T
    audioSfreq = float(streams[1]["info"]["nominal_srate"][0])

    audioInfo = mne.create_info(audioStreamName, audioSfreq, ["misc"])
    audioRaw = mne.io.RawArray(audio, audioInfo)


    ## Preprocessing and envelope
    audioRaw.apply_function(lambda x:  mne.baseline.rescale(data=x, times=audioRaw.times, baseline=(-1,1)), picks="all")

    # Equalize length
    #print("HONK", eegRaw.first_time, audioRaw.first_time)

    htAudio = audioRaw.copy()
    htAudio.apply_hilbert(picks="all")

    envAudio = htAudio
    envAudio.apply_function(np.abs, dtype=np.dtype(np.float64), picks="all")

    fEnvAudio = envAudio
    fEnvAudio.filter(128, None, picks="all")
    fEnvAudio.apply_function(lambda x: np.mean(x, axis=0, keepdims=True), channel_wise=False, picks="all")
    fEnvAudio.apply_function(lambda x: mne.baseline.rescale(data=x, times=audioRaw.times, baseline=(0,1)), picks="all")

    ## We need to reshape the data (downsampled) so we need to create the RawArray from scratch.
    #dsFEnvAudio = np.interp(x=eegRaw.times, xp=audioRaw.times, fp=audioRaw.get_data()[0])
    dsFEnvAudio = fEnvAudio
    dsFEnvAudio = dsFEnvAudio.resample(eegSfreq, events=eegRaw.get_data(picks="all"))[0]


    ## TRF calculation
    uEEG = eegRaw

    uEEG.crop(tmax=dsFEnvAudio.times[-1])
    uEEG.add_channels([dsFEnvAudio])
    uEEG.filter(0.1, 15, picks="eeg")


    fEnvAudio.plot(scalings=None, duration=2, start=0, block=True)
    eegRaw.plot(scalings=dict(eeg=1e-6), duration=2, start=0, block=True)


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
