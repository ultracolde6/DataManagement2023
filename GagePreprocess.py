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

# TODO: Let's build up a front panel for the gage parameters
# TODO: Fix up the file storage scheme and folder crawling scheme
# TODO: Develop method to shuttle files to external drive
# TODO: Set up default and user processing options
# TODO: Set up automated background processing


PHOTON_ENERGY = 2.55e-19
PHOTONRATE_CONVERSION = 1e-6/(2e7)/ PHOTON_ENERGY / (0.5*0.8) # count/us
VOLTAGE_CONVERSION = 1000/32768 #in units of mV

class GagePreprocessor:
    def __init__(self, 
                 run_name, 
                 filter_time, 
                 step_time, 
                 time_me, 
                 plot_bool,
                 het_freq,
                 dds_freq,
                 samp_freq,
                 kappa,
                 LO_power):
        self.run_name = run_name
        self.filter_time = filter_time
        self.step_time = step_time
        self.time_me = time_me
        self.plot_bool = plot_bool
        self.now = 0
        self.start_time = 0
        self.het_freq = het_freq
        self.dds_freq = dds_freq
        self.samp_freq = samp_freq
        self.kappa = kappa
        self.LO_power = LO_power
        
        self.LO_rate = 1e-12 * self.LO_power / PHOTON_ENERGY # count/us
        #2e7 is the conversion gain, 2.55e-19 is single photon energy, the rest is the path efficiency
        self.heterodyne_conversion = 1/np.sqrt(self.LO_rate) # np.sqrt(count/us)
        self.cavity_conversion = 1/np.sqrt(self.kappa)
        self.conversion_factor = VOLTAGE_CONVERSION*PHOTON_ENERGY*self.heterodyne_conversion*self.cavity_conversion

        self.data_path_gage = Path('c:/Users/tyxia/Documents/UltracoldRepos/DataManagement2023/')
        self.file_prefix_gage = 'gage_shot'
        # path, dirs, files = next(os.walk(data_path_gage))
        num_shot_gage_start = 0
        # self.num_shots_gage = len(files)
        self.num_shots_gage = 1
        self.file_reading_time = 0
        self.inner_product_time = 0

    def load_file(self, start):
        """
        Load Gage data files for a given run and filter time.

        Args:
            self.start_time (float): The start time of the data acquisition.
            self.run_name (str): The name of the run.
            self.filter_time (float): The filter time used for the data acquisition.
            self.step_time (float): The step time used for the data acquisition.
            self.time_me (bool): Whether or not to time the file reading and inner product calculation.

        Returns:
            tuple: A tuple containing the number of shots loaded, file reading time, inner product time, and the current time.
        """
        self.start_time = start
        try:
            num_shots_loaded = np.load(f'{self.run_name}_gage_cmplx_amp_{self.filter_time}_{self.step_time}.pkl', allow_pickle=True).shape[1]
        except FileNotFoundError:
            num_shots_loaded = 0
        
        try:
            np.load(f'{self.run_name}_gage_timebin_{self.filter_time}_{self.step_time}.pkl', allow_pickle=True)
        except FileNotFoundError:
            num_shots_loaded = 0
        
        if self.time_me:
            self.file_reading_time = 0
            self.inner_product_time = 0
            self.now = self.start_time

        return num_shots_loaded
    
    def gen_vectors(self, hf):
        """
        Generate vectors for signal processing.

        Parameters:
        hf (h5py.File): An open HDF5 file.

        Returns:
        tuple: Returns a tuple containing:
            - ch1_pure_vec (np.array): The pure vector for channel 1.
            - ch3_pure_vec (np.array): The pure vector for channel 3.
            - t0_list (np.array): The list of initial times.
            - timebin_array (np.array): The array of time bins.
            - cmplx_amp_array (np.array): The array of complex amplitudes.
            - t0_idx_list (np.array): The list of initial times in units of samples.
        """

        # create time vector
        chlen = len(np.array(hf.get('CH1')))
        # compression happens: length is divided by sampling frequency
        t_vec = np.arange(chlen) * 1 / self.samp_freq

        # create pure vectors for channels 1 and 3
        ch1_pure_vec = np.exp(-1j * 2 * np.pi * self.dds_freq * t_vec)
        ch3_pure_vec = np.exp(-1j * 2 * np.pi * self.het_freq * t_vec)

        # create time bin array
        t0_list = np.arange(0, chlen / self.samp_freq - self.filter_time + self.step_time, self.step_time)
        timebin_array = np.empty((len(t0_list), 2), dtype=float)
        timebin_array[:, 0] = t0_list
        timebin_array[:, 1] = t0_list + self.filter_time

        # create complex amplitude array, store complex amplitudes of signal for each shot and time bin
        cmplx_amp_array = np.empty((2, self.num_shots_gage, len(t0_list)), dtype=np.cdouble)

        # create list of initial times in units of samples
        t0_idx_list = np.arange(0, chlen - self.filter_time * self.samp_freq + 1, self.step_time * self.samp_freq)

        return ch1_pure_vec, ch3_pure_vec, t0_list, timebin_array, cmplx_amp_array, t0_idx_list
    
    
    def plot_tenth_shot(self,
                        cmplx_amp_list_ch1, 
                        cmplx_amp_list_ch3, 
                        ch1, 
                        ch3, 
                        t0_list):
        """
        Plot the tenth shot.

        Parameters:
        cmplx_amp_list_ch1 (list): The list of complex amplitudes for channel 1.

        Returns:
        None
        """
        # plot tenth shot
        t_start = 200
        amp_list_ch1 = abs(cmplx_amp_list_ch1)
        angle_list_ch1 = np.angle(cmplx_amp_list_ch1)%(2*np.pi)
        amp_list_ch3 = abs(cmplx_amp_list_ch3)
        angle_list_ch3 = np.angle(cmplx_amp_list_ch3)%(2*np.pi)
        fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(16,5))
        ax1.plot(np.arange(0,self.filter_time,1/self.samp_freq), ch1[0:self.filter_time*self.samp_freq])
        ax2.plot(np.arange(0,self.filter_time,1/self.samp_freq), ch3[0:self.filter_time*self.samp_freq])
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
        
    
    def gage_transforms(self,
                     ch1_pure_vec,
                     ch3_pure_vec,
                     t0_list,
                     t0_idx_list,
                     cmplx_amp_array):
        """
        Applies various transformations to Gage data.

        Args:
        - file_reading_time (float): time taken to read the file
        - inner_product_time (float): time taken to calculate the inner product
        - now (float): current time
        - num_shots_gage (int): number of shots in the Gage data
        - self.filter_time (float): time for filtering
        - self.step_time (float): time for stepping
        - self.time_me (bool): whether to measure time or not
        - ch1_pure_vec (numpy array): array for channel 1
        - ch3_pure_vec (numpy array): array for channel 3
        - t0_list (numpy array): list of time bins
        - t0_idx_list (numpy array): list of indices for time bins
        - cmplx_amp_array (numpy array): array for complex amplitudes
        - plot_tenth_shot (function): function to plot the tenth shot

        Returns:
        - cmplx_amp_array (numpy array): array for complex amplitudes
        - file_reading_time (float): time taken to read the file
        - inner_product_time (float): time taken to calculate the inner product
        - now (float): current time
        """
        for shot_num in range(self.num_shots_gage):
            if shot_num%100 == 0:
                print(f'shot{shot_num} done')
            #mask_valid_data[shot_num]:
            # construct file name
            file_name_gage = self.file_prefix_gage+'_'+str(shot_num).zfill(5)+'.h5'
            hf = h5py.File(self.data_path_gage/file_name_gage, 'r')
            
            # read data and apply conversion factor
            ch1 = np.array(hf['CH1']) * self.conversion_factor #np.sqrt(n)
            ch3 = np.array(hf['CH3']) * self.conversion_factor #np.sqrt(n)

            if self.time_me:
                self.file_reading_time += time.time() - self.now
                self.now = time.time()

            # demodulate data and center frequency at zero
            ch1_demod = np.multiply(ch1 , ch1_pure_vec) # * 1j
            ch3_demod = np.multiply(ch3 , ch3_pure_vec) # * 1j
            ch1_demod_sum = np.concatenate([[0],np.cumsum(ch1_demod)])
            ch3_demod_sum = np.concatenate([[0],np.cumsum(ch3_demod)])
            cmplx_amp_list_ch1=t0_list*0j
            cmplx_amp_list_ch3=t0_list*0j

            # calculate complex amplitude for each time bin
            for i,t0_idx in enumerate(t0_idx_list):
                t0_idx = int(t0_idx)
                t1_idx = int(t0_idx + self.filter_time*self.samp_freq)
                cmplx_amp_list_ch1[i] = (ch1_demod_sum[t1_idx] - ch1_demod_sum[t0_idx]) / (t1_idx-t0_idx)
                cmplx_amp_list_ch3[i] = (ch3_demod_sum[t1_idx] - ch3_demod_sum[t0_idx]) / (t1_idx-t0_idx)

        if self.time_me:
            self.inner_product_time += time.time() - self.now
            self.now = time.time()

        # store complex amplitudes in array
        cmplx_amp_array[0, shot_num] = cmplx_amp_list_ch1
        cmplx_amp_array[1, shot_num] = cmplx_amp_list_ch3

        # plot tenth shot
        if self.plot_bool and shot_num == 50:
            plot_tenth_shot(cmplx_amp_list_ch1,
                            cmplx_amp_list_ch3,
                            ch1,
                            ch3,
                            t0_list)

        return cmplx_amp_array
    
    
    def preprocess_gage(self):
        print(f'Hard reset, loading {self.num_shots_gage} shots from gage raw data')
        # construct file name
        file_name_gage = self.file_prefix_gage+'_00000.h5'
        hf = h5py.File(self.data_path_gage/file_name_gage, 'r')
        # store time and frequency vectors after compression
        ch1_pure_vec, ch3_pure_vec, t0_list, timebin_array, cmplx_amp_array, t0_idx_list = self.gen_vectors(hf)

        cmplx_amp_array = self.gage_transforms(
                                        ch1_pure_vec,
                                        ch3_pure_vec,
                                        t0_list,
                                        t0_idx_list,
                                        cmplx_amp_array)

        return cmplx_amp_array, timebin_array

    
    def gage_loop(self):
        start = time.time()
        num_shots_loaded = self.load_file(start)
        cmplx_amp_array, timebin_array = self.preprocess_gage()
        
        # save files to pickle files
        with open(f'{self.run_name}_gage_cmplx_amp_{self.filter_time}_{self.step_time}.pkl','wb') as f1:
            pickle.dump(cmplx_amp_array, f1)
        with open(f'{self.run_name}_gage_timebin_{self.filter_time}_{self.step_time}.pkl','wb') as f3:
            pickle.dump(timebin_array, f3)

        print('done')
        print(f"number of {self.num_shots_gage} shots loaded")
        print(f"total time elapsed {time.time()-self.start_time} s")
        

# %%
########################### User input section ############################
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
time_me = True
plot_tenth_shot = True

het_freq = 20.000446 #MHz
dds_freq = het_freq/2
samp_freq = 200 #MHz
step_time = 50 #us
filter_time = 100 #us

PHOTON_ENERGY = 2.55e-19
photonrate_conversion = 1e-6/(2e7)/ PHOTON_ENERGY / (0.5*0.8) # count/us
voltage_conversion = 1000/32768 #in units of mV
kappa = 2*np.pi * 1.1 #MHz
LO_power = 314 #uW
LO_rate = 1e-12 * LO_power / PHOTON_ENERGY # count/us
#2e7 is the conversion gain, 2.55e-19 is single photon energy, the rest is the path efficiency
heterodyne_conversion = 1/np.sqrt(LO_rate) # np.sqrt(count/us)

cavity_conversion = 1/np.sqrt(kappa)
conversion_factor = voltage_conversion*photonrate_conversion*heterodyne_conversion*cavity_conversion

datastream_name_gage='gage'
working_path_gage = Path.cwd().parent
data_path_gage = working_path_gage/'DataManagement2023'
file_prefix_gage = 'gage_shot'
path, dirs, files = next(os.walk(data_path_gage))
num_shot_gage_start = 0
num_shots_gage = len(files)
if override_num_shots:
    num_shots_gage = num_shots_manual


gage_obj = GagePreprocessor(run_name, 
                            filter_time, 
                            step_time, 
                            time_me, 
                            plot_tenth_shot, 
                            het_freq, 
                            dds_freq, 
                            samp_freq, 
                            kappa, 
                            LO_power)

gage_obj.gage_loop()
# %%
