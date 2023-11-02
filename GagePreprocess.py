# %%
from pathlib import Path
import numpy as np 
import h5py
import matplotlib.pyplot as plt
import time
import os
from scipy import signal
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit
from scipy.special import erf
import scipy.io
import colorsys
from scipy import signal
import math
from scipy.stats import norm
import pickle
import os
import pandas as pd
import ipyparams


# %%
########################### User input section ############################
currentnotebook_name = ipyparams.notebook_name
run_name = currentnotebook_name[:-6]
run_name = 'run0'

#Boolean input
override_num_shots = False
reset_hard = False #only set this to true if you want to reload from raw data

#number input
num_shots_manual = 500
num_frames = 3
outer_zoom_factor = 5

point_name_outer = 'sideprobe_intensity'
point_name_inner = 'cav_pzt'

point_list_outer = np.array([1.7]) #ControlV_Power(np.array([1,1.2,1.4,1.6,1.8,2,2.5,4]))
point_list_inner = np.linspace(-0.6,0.6,7)

tweezer_freq_list = 88 + 0.8*0 + 0.8*np.arange(40)
twz_num_plot=np.array([18])
atom_site = []
for i in range(num_frames):
    atom_site.append(np.arange(len(tweezer_freq_list)))

#################################################################################
num_points_inner = len(point_list_inner)
num_points_outer = len(point_list_outer)

if num_points_outer == 1:
    point_list = np.array(point_list_inner)
else:
    point_list = (outer_zoom_factor*np.outer(point_list_outer,np.ones(len(point_list_inner))) + np.outer(np.ones(len(point_list_outer)),point_list_inner)).flatten()
    plt.plot(point_list)
num_points = len(point_list)

# gagescope parameters
# %%
reset_gage = reset_hard
# reset_gage = True
time_me = True
plot_tenth_shot = True
het_freq = 20.000446 #MHz
# het_freq = 20.000 #MHz
dds_freq = het_freq/2
samp_freq = 200 #MHz
# averaging_time = 100 #us
# step_time = 5 #us
# filter_time = 10 #us
step_time = 50 #us
filter_time = 100 #us
voltage_conversion = 1000/32768 #in units of mV
kappa = 2*np.pi * 1.1 #MHz
LO_power = 314 #uW
PHOTON_ENERGY = 2.55e-19
LO_rate = 1e-12 * LO_power / PHOTON_ENERGY # count/us
photonrate_conversion = 1e-6/(2e7)/ PHOTON_ENERGY / (0.5*0.8) # count/us
#2e7 is the conversion gain, 2.55e-19 is single photon energy, the rest is the path efficiency
heterodyne_conversion = 1/np.sqrt(LO_rate) # np.sqrt(count/us)
cavity_conversion = 1/np.sqrt(kappa)
conversion_factor = voltage_conversion*photonrate_conversion*heterodyne_conversion*cavity_conversion
# filter_type = "square"
datastream_name_gage='gage'
working_path_gage = Path.cwd().parent
data_path_gage = working_path_gage/'DataManagement2023'
file_prefix_gage = 'gage_shot'
path, dirs, files = next(os.walk(data_path_gage))
num_shot_gage_start = 0
num_shots_gage = len(files)
if override_num_shots:
    num_shots_gage = num_shots_manual


# %%
# start processing
def load_file(start_time, run_name, filter_time, step_time, time_me):
    try:
        num_shots_loaded = np.load(f'{run_name}_gage_cmplx_amp_{filter_time}_{step_time}.pkl', allow_pickle=True).shape[1]
    except FileNotFoundError:
        num_shots_loaded = 0
    try:
        np.load(f'{run_name}_gage_timebin_{filter_time}_{step_time}.pkl', allow_pickle=True)
    except FileNotFoundError:
        num_shots_loaded = 0
    if time_me:
        file_reading_time = 0
        inner_product_time = 0
        now = start_time

    return num_shots_loaded, file_reading_time, inner_product_time, now


def gen_vectors(hf):
    # create time vector
    chlen = len(np.array(hf.get('CH1')))
    # compression happens: length is divided by sampling frequency
    t_vec = np.arange(chlen)*1/samp_freq
    ch1_pure_vec = np.exp(-1j*2*np.pi * dds_freq*t_vec)
    ch3_pure_vec = np.exp(-1j*2*np.pi * het_freq*t_vec)
    # create time bin array
    t0_list = np.arange(0,chlen/samp_freq-filter_time+step_time,step_time)
    # store step-time array
    timebin_array = np.empty((len(t0_list),2), dtype=float)
    timebin_array[:,0] = t0_list
    timebin_array[:,1] = t0_list + filter_time
    # create complex amplitude array, store complex amplitudes of signal for each shot and time bin
    cmplx_amp_array = np.empty((2, num_shots_gage,len(t0_list)), dtype=np.cdouble)
    # create list of initial times in units of samples
    t0_idx_list = np.arange(0,chlen-filter_time*samp_freq+1,step_time*samp_freq)

    return ch1_pure_vec, ch3_pure_vec, t0_list, timebin_array, cmplx_amp_array, t0_idx_list


def plot_tenth_shot(cmplx_amp_list_ch1, 
                    cmplx_amp_list_ch3, 
                    ch1, 
                    ch3, 
                    t0_list, 
                    filter_time, 
                    samp_freq):
    # plot tenth shot
    t_start = 200
    amp_list_ch1 = abs(cmplx_amp_list_ch1)
    angle_list_ch1 = np.angle(cmplx_amp_list_ch1)%(2*np.pi)
    amp_list_ch3 = abs(cmplx_amp_list_ch3)
    angle_list_ch3 = np.angle(cmplx_amp_list_ch3)%(2*np.pi)
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(16,5))
    ax1.plot(np.arange(0,filter_time,1/samp_freq), ch1[0:filter_time*samp_freq])
    ax2.plot(np.arange(0,filter_time,1/samp_freq), ch3[0:filter_time*samp_freq])
    ax1.set_ylabel("mV")
    ax2.set_ylabel("mV")
    plt.show()
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(16,5))
    ax1.plot(t0_list,(2*np.unwrap(angle_list_ch1)))
    ax1.plot(t0_list,np.unwrap(angle_list_ch3))
    ax2.plot(t0_list,amp_list_ch1)
    ax2.plot(t0_list,amp_list_ch3)
    ax1.set_xlabel("us")
    ax2.set_xlabel("us")
    plt.show()


def apply_transforms(file_reading_time, 
                     inner_product_time, 
                     now, num_shots_gage, 
                     filter_time, 
                     step_time, 
                     time_me,
                     ch1_pure_vec,
                     ch3_pure_vec,
                     t0_list,
                     t0_idx_list,
                     cmplx_amp_array,
                     plot_tenth_shot):
    for shot_num in range(num_shots_gage):
        if shot_num%100 == 0:
            print(f'shot{shot_num} done')
        #mask_valid_data[shot_num]:
        # construct file name
        file_name_gage = file_prefix_gage+'_'+str(shot_num).zfill(5)+'.h5'
        hf = h5py.File(data_path_gage/file_name_gage, 'r')
        
        # read data and apply conversion factor
        ch1 = np.array(hf['CH1']) * conversion_factor #np.sqrt(n)
        ch3 = np.array(hf['CH3']) * conversion_factor #np.sqrt(n)

        if time_me:
            file_reading_time += time.time() - now
            now = time.time()

        # demodulate data and center frequency at zero
        ch1_demod = np.multiply(ch1 , ch1_pure_vec) # * 1j
        ch3_demod = np.multiply(ch3 , ch3_pure_vec) # * 1j
        ch1_demod_sum = np.concatenate([[0],np.cumsum(ch1_demod)])
        ch3_demod_sum = np.concatenate([[0],np.cumsum(ch3_demod)])
        cmplx_amp_list_ch1=t0_list*0j
        cmplx_amp_list_ch3=t0_list*0j

        # calculate complex amplitude for each time bin
        for i,t0_idx in enumerate(t0_idx_list):
            t1_idx = t0_idx + filter_time*samp_freq
    #                 cmplx_amp_list_ch1[i] = np.mean(ch1_demod[t0_idx:t1_idx])
    #                 cmplx_amp_list_ch3[i] = np.mean(ch3_demod[t0_idx:t1_idx])
            cmplx_amp_list_ch1[i] = (ch1_demod_sum[t1_idx] - ch1_demod_sum[t0_idx]) / (t1_idx-t0_idx)
            cmplx_amp_list_ch3[i] = (ch3_demod_sum[t1_idx] - ch3_demod_sum[t0_idx]) / (t1_idx-t0_idx)

        if time_me:
            inner_product_time += time.time() - now
            now = time.time()

        # store complex amplitudes in array
        cmplx_amp_array[0, shot_num] = cmplx_amp_list_ch1
        cmplx_amp_array[1, shot_num] = cmplx_amp_list_ch3

        # plot tenth shot
        if plot_tenth_shot and shot_num == 50:
            plot_tenth_shot(cmplx_amp_list_ch1,
                            cmplx_amp_list_ch3,
                            ch1,
                            ch3,
                            t0_list,
                            filter_time,
                            samp_freq)

        return cmplx_amp_array, file_reading_time, inner_product_time, now


def preprocess_gage(file_reading_time, 
                    inner_product_time, 
                    now, 
                    num_shots_gage, 
                    num_shots_loaded, 
                    filter_time, 
                    step_time, 
                    time_me,
                    plot_tenth_shot):
    print(f'Hard reset, loading {num_shots_gage} shots from gage raw data')
    # construct file name
    file_name_gage = file_prefix_gage+'_00000.h5'
    hf = h5py.File(data_path_gage/file_name_gage, 'r')
    # store time and frequency vectors after compression
    ch1_pure_vec, ch3_pure_vec, t0_list, timebin_array, cmplx_amp_array, t0_idx_list = gen_vectors(hf)

    cmplx_amp_array, file_reading_time, inner_product_time, now = apply_transforms(file_reading_time,
                                                                                    inner_product_time,
                                                                                    now,
                                                                                    num_shots_gage,
                                                                                    filter_time,
                                                                                    step_time,
                                                                                    time_me,
                                                                                    ch1_pure_vec,
                                                                                    ch3_pure_vec,
                                                                                    t0_list,
                                                                                    t0_idx_list,
                                                                                    cmplx_amp_array,
                                                                                    plot_tenth_shot)

    return file_reading_time, inner_product_time, now, cmplx_amp_array, timebin_array


def main():
    start = time.time()
    num_shots_loaded, file_reading_time, inner_product_time, now = load_file(start, run_name, filter_time, step_time, time_me)
    file_reading_time,inner_product_time, now, cmplx_amp_array, timebin_array = preprocess_gage(file_reading_time, inner_product_time, now, num_shots_gage, num_shots_loaded, filter_time, step_time, time_me, plot_tenth_shot)
    
    # save files to pickle files
    with open(f'{run_name}_gage_cmplx_amp_{filter_time}_{step_time}.pkl','wb') as f1:
        pickle.dump(cmplx_amp_array, f1)
    with open(f'{run_name}_gage_timebin_{filter_time}_{step_time}.pkl','wb') as f3:
        pickle.dump(timebin_array, f3)

    print('done')
    print(f"number of {num_shots_gage} shots loaded")
    print(f"total time elapsed {time.time()-start} s")


# %%
main()

# %%
if __name__ == '__main__':
    main()
