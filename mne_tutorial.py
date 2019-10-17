import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_response(signal, argument):
    """plot response to check what happened with the data"""
    if "time" in argument:
        signal.plot(duration=10, remove_dc=False)
    if "psd" in argument:
        signal.plot_psd(fmin=0, fmax=80)
    if "butter" in argument:
        signal.plot(butterfly=True, color='#00000044', bad_color='r')


def detect_bad_ch(eeg):
    """plots each channel so user can decide whether good (mouse click) or bad (enter / space)"""
    good_ch, bad_ch = [], []
    intvl = eeg.__len__() // 20

    for ch in eeg.ch_names:
        """loop over each channel and plot to decide if bad"""
        time_data = eeg[eeg.ch_names.index(ch)][0][0]
        df = pd.DataFrame()

        for i in range(20):
            df_window = pd.DataFrame(time_data[i * intvl:(i + 1) * intvl])
            df_window += (i + 1) * 0.0001
            df = pd.concat((df, df_window), axis=1)

        df *= 1000  # just for plotting
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f"{ch}: mouse click for keep (good), any other key for remove (bad)")
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=3)
        ax1.psd(time_data, new_sampling, new_sampling)
        ax1.set_xlim([0, 100])
        ax2.plot(df, 'b')
        plt.show()

        if not plt.waitforbuttonpress():
            good_ch.append(ch)
            plt.close(fig)
        else:
            bad_ch.append(ch)
            plt.close(fig)

    return good_ch, bad_ch


# define path and filename
filename = "sleep_sd1020_eyes_closed2_060219.vhdr"
filepath = Path("/home/benjamin/Downloads/sd1020/")
file = filepath / filename


# 1. load data
data = mne.io.read_raw_brainvision(file)
data.load_data()
# plot_response(data, 'time')


# 2. TMS-artifact interpolation!!!!


# 3. resample (with low-pass filter)
new_sampling = 500
data_sampled = data.copy().resample(new_sampling, npad='auto')
# plot_response(data_sampled, ['time', 'psd'])


# 4. filter (first high- then low-pass)
l_cut, h_cut = 0.5, 40
data_filtered = data_sampled.copy().filter(l_freq=l_cut, h_freq=h_cut)
# plot_response(data_filtered, 'psd')


# 5. channel info (remove EMG and set type for EOG channels)
data_filtered.drop_channels('EMG1')
data_filtered.set_channel_types({'VEOGl': 'eog', 'VEOGu': 'eog'})
data_channel = data_filtered.copy().set_montage('standard_1005')


# 6. remove bad channels (or do not remove but track them)
good, bad = detect_bad_ch(data_channel)
data_channel.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
# data_channel = data_channel.copy().drop_channels(bad)  # remove bad channels (eeglab style)


# 7. interpolate channels (spherical spline method recommended)
data_interp = data_channel.copy().interpolate_bads(reset_bads=True)  # for presentation of bad channels change to False
plot_response(data_interp, 'butter')


# 8. re-reference to average
data_reref = data_interp.copy().set_eeg_reference('average', projection=False)  # you might want to go with True


# 9. PCA (optional)


# 10. ICA


"""create sine wave for fun"""
# import numpy as np
# import matplotlib.pyplot as plt
#
# A = 2
# f = 10
# s = 100
# t = np.arange(0, 1, 1/s)
# ph = 0.25
#
# x = A*np.sin(2*np.pi*f*t+ph)
# plt.plot(t, x)
# plt.show()
