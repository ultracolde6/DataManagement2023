#!/usr/bin/python
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
from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm


# TODO: There is some issue with the pickle running

# %%

# %%
def gen_power_curve(x_int, y_int):
    y_int = y_int / np.max(y_int)
    power_curve = interp1d(x_int, y_int, kind='slinear')
    plt.plot(power_curve(x_int))
    return power_curve


def ControlV_Power(x, maximum_power=1):
    x_int = np.array(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9, 2, 2.2, 2.5, 2.75,
         3, 3.25, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10])
    y_int = np.array(
        [0.01, 0.05, 0.37, 1.61, 4.6, 10.33, 19.55, 32.62, 49.72, 70.12, 93.72, 119.12, 145.82, 174.32, 203.32, 232.32,
         288.52, 311.92,
         335.92, 355.92, 387.92, 411.92, 421.92, 428.92, 430.92, 433.92, 437.92, 441.92, 442.92, 445.92, 447.92, 449.92,
         449.92, 451.42])
    power_curve = gen_power_curve(x_int, y_int)

    return maximum_power * power_curve(x)


def get_outer_inner_point_product(point_idx, outer_num, inner_num):
    total_num = outer_num * inner_num
    if point_idx >= total_num:
        print('point_idx is out of range')
        return
    outer_point_idx = point_idx // inner_num
    inner_point_idx = point_idx % inner_num
    return outer_point_idx, inner_point_idx


def get_outer_inner_point_parallel(point_idx, outer_num, inner_num):
    total_num = outer_num * inner_num
    if point_idx >= total_num:
        print('point_idx is out of range')
        return
    outer_point_idx = point_idx % outer_num
    inner_point_idx = point_idx % inner_num
    return outer_point_idx, inner_point_idx


def get_outer_inner_point(point_idx, outer_num, inner_num, parallel=False):
    if parallel:
        return get_outer_inner_point_parallel(point_idx, outer_num, inner_num)
    else:
        return get_outer_inner_point_product(point_idx, outer_num, inner_num)


# %%
# TODO: HOW TO BEST LET USER SET CONSTANTS? Maybe just put dict in analysis nb
def gen_jkam_constants(user_constants_dict,
                       jkam_dict,
                       outer_zoom_factor,
                       atom_site,
                       num_points_inner,
                       num_points_outer,
                       point_list,
                       num_points):
    run_name = user_constants_dict['run_name']
    override_num_shots = user_constants_dict['override_num_shots']
    reset_hard = user_constants_dict['reset_hard']
    num_shots_manual = user_constants_dict['num_shots_manual']
    num_frames = user_constants_dict['num_frames']
    point_name_inner = user_constants_dict['point_name_inner']
    point_name_outer = user_constants_dict['point_name_outer']
    point_list_outer = user_constants_dict['point_list_outer']
    point_list_inner = user_constants_dict['point_list_inner']
    point_parallel = user_constants_dict['point_parallel']
    tweezer_freq_list = user_constants_dict['tweezer_freq_list']

    jkam_constants = {'run_name': run_name,
                      'override_num_shots': override_num_shots,
                      'reset_hard': reset_hard,
                      'num_shots_manual': num_shots_manual,
                      'num_frames': num_frames,
                      'point_name_inner': point_name_inner,
                      'point_name_outer': point_name_outer,
                      'point_list_outer': point_list_outer,
                      'point_list_inner': point_list_inner,
                      'point_parallel': point_parallel,
                      'outer_zoom_factor': outer_zoom_factor,
                      'tweezer_freq_list': tweezer_freq_list,
                      'atom_site': atom_site,
                      'num_points_inner': num_points_inner,
                      'num_points_outer': num_points_outer,
                      'point_list': point_list,
                      'num_points': num_points}

    return jkam_constants


# %%
def gen_gage_constants(arb_constant_dict, gage_constant_dict, jkam_constant_dict, date_dir):
    jkam_constants = jkam_constant_dict
    dir_data = ('X:/expdata-e6/data')
    outpath = dir_data + '/' + date_dir + '/analysis'
    print("reading into files at: ", outpath)
    reset_gage = gage_constant_dict['reset_gage']
    window = gage_constant_dict['window']
    num_segments = gage_constant_dict['num_segments']
    plot_tenth_shot = gage_constant_dict['plot_tenth_shot']
    het_freq = gage_constant_dict['het_freq']
    dds_freq = gage_constant_dict['dds_freq']
    samp_freq = gage_constant_dict['samp_freq']
    step_time = gage_constant_dict['step_time']
    filter_time = gage_constant_dict['filter_time']
    datastream_name_gage = gage_constant_dict['datastream_name']
    working_path_gage = gage_constant_dict['working_path']
    run_name = gage_constant_dict['run_name']
    file_prefix_gage = gage_constant_dict['file_prefix']

    voltage_conversion = arb_constant_dict['voltage_conversion']
    kappa = arb_constant_dict['kappa']
    LO_power = arb_constant_dict['LO_power']
    PHOTON_ENERGY = arb_constant_dict['PHOTON_ENERGY']

    LO_rate = 1e-12 * LO_power / PHOTON_ENERGY
    photonrate_conversion = 1e-6 / (4.64e7) / (22.36) / PHOTON_ENERGY / (0.5 * 0.8)  # count/us
    heterodyne_conversion = 1 / np.sqrt(LO_rate)  # np.sqrt(count/us)
    cavity_conversion = 1 / np.sqrt(kappa)
    conversion_factor = voltage_conversion * photonrate_conversion * heterodyne_conversion * cavity_conversion

    data_path_gage = working_path_gage / 'data' / run_name / datastream_name_gage
    file_prefix_gage = 'gage_shot'
    path, dirs, files = next(os.walk(data_path_gage))

    num_shot_gage_start = 0
    num_shots_gage = len(files)
    override_num_shots = jkam_constants['override_num_shots']
    num_shots_manual = jkam_constants['num_shots_manual']

    if override_num_shots:
        num_shots_gage = num_shots_manual

    start = time.time()
    try:
        num_shots_loaded = \
            np.load(outpath + '\\' + f'{run_name}_{window}_gage_cmplx_amp_{filter_time}_{step_time}.pkl',
                    allow_pickle=True).shape[1]

        print("file found in folder, we can unpickle the pickle: ", outpath)
    except FileNotFoundError:
        num_shots_loaded = 0
        print("file NOT found in folder, we are pickleless :(: ", outpath)
    try:
        np.load(outpath + '\\' + f'{run_name}_{window}_gage_timebin_{filter_time}_{step_time}.pkl', allow_pickle=True)
    except FileNotFoundError:
        num_shots_loaded = 0

    gage_constant_dict = {'reset_gage': reset_gage,
                          'window': window,
                          'num_segments': num_segments,
                          'plot_tenth_shot': plot_tenth_shot,
                          'het_freq': het_freq,
                          'dds_freq': dds_freq,
                          'samp_freq': samp_freq,
                          'step_time': step_time,
                          'filter_time': filter_time,
                          'voltage_conversion': voltage_conversion,
                          'kappa': kappa,
                          'LO_power': LO_power,
                          'PHOTON_ENERGY': PHOTON_ENERGY,
                          'LO_rate': LO_rate,
                          'photonrate_conversion': photonrate_conversion,
                          'heterodyne_conversion': heterodyne_conversion,
                          'cavity_conversion': cavity_conversion,
                          'conversion_factor': conversion_factor,
                          'datastream_name': datastream_name_gage,
                          'working_path': working_path_gage,
                          'run_name': run_name,
                          'data_path': data_path_gage,
                          'file_prefix': file_prefix_gage,
                          'num_shot_gage_start': num_shot_gage_start,
                          'num_shots': num_shots_gage,
                          'num_shots_loaded': num_shots_loaded,
                          'file_suffix': '.h5',
                          'date_dir': date_dir}

    return gage_constant_dict


# %%
def gen_jkam_mask_info(jkam_dict, jkam_constants):
    arb_constants = jkam_constants
    run_name = arb_constants['run_name']
    override_num_shots = arb_constants['override_num_shots']
    reset_hard = arb_constants['reset_hard']
    num_shots_manual = arb_constants['num_shots_manual']
    num_frames = arb_constants['num_frames']
    point_name_inner = arb_constants['point_name_inner']
    point_name_outer = arb_constants['point_name_outer']
    point_list_outer = arb_constants['point_list_outer']
    point_list_inner = arb_constants['point_list_inner']
    point_parallel = arb_constants['point_parallel']
    outer_zoom_factor = arb_constants['outer_zoom_factor']
    tweezer_freq_list = arb_constants['tweezer_freq_list']
    atom_site = arb_constants['atom_site']
    num_points_inner = arb_constants['num_points_inner']
    num_points_outer = arb_constants['num_points_outer']
    point_list = arb_constants['point_list']
    num_points = arb_constants['num_points']

    datastream_name = jkam_dict['datastream_name']
    working_path = jkam_dict['working_path']
    data_path = working_path / 'data' / run_name / datastream_name
    file_prefix = jkam_dict['file_prefix']
    path, dirs, files = next(os.walk(data_path))

    num_shots_jkam = len(files)
    num_shots = num_shots_jkam
    num_loops = num_shots // num_points
    num_tweezers = len(tweezer_freq_list)
    num_points = len(point_list)
    num_loops = num_shots // num_points + bool(num_shots % num_points)
    print(f'num_points={num_points}, num_loops={num_loops}')
    print(f'num_shots={num_shots}')

    if override_num_shots:
        num_shots = num_shots_manual

    jkam_creation_time_array = np.zeros(num_shots)
    print("Gathering JKAM creation times...")
    for shot_num in range(num_shots):
        file_name = file_prefix + '_' + str(shot_num).zfill(5) + '.h5'
        # jkam_creation_time_array[shot_num] = os.path.getctime(data_path/file_name)
        jkam_creation_time_array[shot_num] = os.path.getmtime(data_path / file_name)
        # get modified time as I am copying files over for testing

    avg_time_gap = (jkam_creation_time_array[-1] - jkam_creation_time_array[0]) / (num_shots - 1)

    jkam_match_dict = {'num_shots_jkam': num_shots_jkam,
                       'num_shots': num_shots,
                       'num_loops': num_loops,
                       'jkam_creation_time_array': jkam_creation_time_array,
                       'avg_time_gap': avg_time_gap}

    return jkam_match_dict


# %%
# # TODO: THIS IS DEPRECATED, USE JKAM_MASK
# def gen_jkamgage_masks(jkam_dict, jkam_constants, gage_constants):
#     jkam_mask_constants = gen_jkam_mask_info(jkam_dict, jkam_constants)
#     num_shots = jkam_mask_constants['num_shots']
#     num_shots_jkam = jkam_mask_constants['num_shots_jkam']
#     num_loops = jkam_mask_constants['num_loops']
#     jkam_creation_time_array = jkam_mask_constants['jkam_creation_time_array']
#     avg_time_gap = jkam_mask_constants['avg_time_gap']

#     num_shots_gage = gage_constants['num_shots']
#     file_prefix_gage = gage_constants['file_prefix']
#     data_path_gage = gage_constants['data_path']

#     gage_creation_time_array = np.zeros(num_shots)
#     print("Gathering Gage creation times...")
#     # progress bar for loading gage data
#     for shot_num in range(num_shots):
#         if shot_num < num_shots_gage:
#             file_name = file_prefix_gage + '_' + str(shot_num).zfill(5) + '.h5'
#             # gage_creation_time_array[shot_num] = os.path.getctime(data_path_gage/file_name)
#             gage_creation_time_array[shot_num] = os.path.getmtime(data_path_gage / file_name)
#             # modified time again
#     # Check data matching
#     mask_valid_data_gage = np.zeros(len(jkam_creation_time_array)) > 1
#     jkam_gage_matchlist = np.zeros(len(jkam_creation_time_array), dtype='int') - 1
#     gage_index_list = np.arange(len(gage_creation_time_array))

#     print("Matching JKAM and Gage data")
#     for shot_num in tqdm(range(num_shots)):
#         time_temp = jkam_creation_time_array[shot_num]
#         space_correct = True
#         if (shot_num > 0) & (np.abs(time_temp - jkam_creation_time_array[
#             shot_num - 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False
#         if (shot_num < (num_shots - 1)):
#             if (np.abs(-time_temp + jkam_creation_time_array[
#                 shot_num + 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False

#         if ((np.min(np.abs(gage_creation_time_array - time_temp)) <= 0.3 * avg_time_gap)) & space_correct:
#             mask_valid_data_gage[shot_num] = True
#             jkam_gage_matchlist[shot_num] = gage_index_list[np.argmin(np.abs(gage_creation_time_array - time_temp))]
#         else:
#             print(f'error at {shot_num:d}')
#             mask_valid_data_gage[shot_num] = False

#     plt.plot(jkam_gage_matchlist)
#     plt.title("JKAM Gagescope file matchlist")
#     plt.xlabel("JKAM shot number")
#     plt.ylabel("Gagescope shot number")

#     return mask_valid_data_gage, jkam_gage_matchlist, gage_index_list


# %%
# TODO: THIS IS UP TO DATE
def gen_jkam_masks(jkam_mask_dict, jkam_dict, jkam_constants, target_constants):
    jkam_mask_constants = jkam_mask_dict
    num_shots = jkam_mask_constants['num_shots']
    num_shots_jkam = jkam_mask_constants['num_shots_jkam']
    num_loops = jkam_mask_constants['num_loops']
    jkam_creation_time_array = jkam_mask_constants['jkam_creation_time_array']
    avg_time_gap = jkam_mask_constants['avg_time_gap']

    num_shots_target = target_constants['num_shots']
    file_prefix_target = target_constants['file_prefix']
    data_path_target = target_constants['data_path']
    file_suffix = target_constants['file_suffix']

    target_creation_time_array = np.zeros(num_shots)
    print("Gathering target device creation times...")
    # progress bar for loading gage data
    for shot_num in range(num_shots):
        if shot_num < num_shots_target:
            file_name = file_prefix_target + '_' + str(shot_num).zfill(5) + file_suffix
            # gage_creation_time_array[shot_num] = os.path.getctime(data_path_gage/file_name)
            target_creation_time_array[shot_num] = os.path.getmtime(data_path_target / file_name)
            # modified time again
    # Check data matching
    mask_valid_data_target = np.zeros(len(jkam_creation_time_array)) > 1
    jkam_gage_matchlist = np.zeros(len(jkam_creation_time_array), dtype='int') - 1
    target_index_list = np.arange(len(target_creation_time_array))

    print("Matching JKAM and target device data")
    for shot_num in tqdm(range(num_shots)):
        time_temp = jkam_creation_time_array[shot_num]
        space_correct = True
        if (shot_num > 0) & (np.abs(time_temp - jkam_creation_time_array[
            shot_num - 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False
        if (shot_num < (num_shots - 1)):
            if (np.abs(-time_temp + jkam_creation_time_array[
                shot_num + 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False

        if ((np.min(np.abs(target_creation_time_array - time_temp)) <= 0.3 * avg_time_gap)) & space_correct:
            mask_valid_data_target[shot_num] = True
            jkam_gage_matchlist[shot_num] = target_index_list[np.argmin(np.abs(target_creation_time_array - time_temp))]
        else:
            print(f'error at {shot_num:d}')
            mask_valid_data_target[shot_num] = False

    plt.plot(jkam_gage_matchlist)
    plt.title("JKAM" + target_constants['datastream_name'] + ' target device file matchlist')
    plt.xlabel("JKAM shot number")
    plt.ylabel(target_constants['datastream_name'] + "target device shot number")

    return mask_valid_data_target, jkam_gage_matchlist, target_index_list


# %%
def RP_jkam_masks(jkam_mask_dict, jkam_dict, jkam_constants, rp_creation_time_array):
    jkam_mask_constants = jkam_mask_dict
    num_shots = jkam_mask_constants['num_shots']
    num_shots_jkam = jkam_mask_constants['num_shots_jkam']
    num_loops = jkam_mask_constants['num_loops']
    jkam_creation_time_array = jkam_mask_constants['jkam_creation_time_array']
    avg_time_gap = jkam_mask_constants['avg_time_gap']

    mask_valid_data_rp = np.zeros(len(jkam_creation_time_array)) > 1
    jkam_rp_matchlist = np.zeros(len(jkam_creation_time_array), dtype='int') - 1
    rp_index_list = np.arange(len(rp_creation_time_array))

    for shot_num in tqdm(range(num_shots)):
        time_temp = jkam_creation_time_array[shot_num]
        space_correct = True
        if (shot_num > 0) & (np.abs(time_temp - jkam_creation_time_array[
            shot_num - 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False
        if (shot_num < (num_shots - 1)):
            if (np.abs(-time_temp + jkam_creation_time_array[
                shot_num + 1] - avg_time_gap) > 0.3 * avg_time_gap): space_correct = False

        if ((np.min(np.abs(rp_creation_time_array - time_temp)) <= 0.3 * avg_time_gap)) & space_correct:
            mask_valid_data_rp[shot_num] = True
            jkam_rp_matchlist[shot_num] = rp_index_list[np.argmin(np.abs(rp_creation_time_array - time_temp))]
        else:
            print(f'error at {shot_num:d}')
            mask_valid_data_rp[shot_num] = False
        if shot_num % 1000 == 0:
            print(shot_num)

    plt.plot(jkam_rp_matchlist)

    return mask_valid_data_rp, jkam_rp_matchlist, rp_index_list


# %%
# TODO: PLOTTING THE SHOTS CAUSES THE PROGRAM TO HANG AT AROUND HALFWAY - DON'T PLOT FOR NOW
def plot_shots(cmplx_amp_list_ch1,
               cmplx_amp_list_ch3,
               ch1,
               ch3,
               t0_list,
               samp_freq,
               filter_time):
    t_start = 200
    amp_list_ch1 = abs(cmplx_amp_list_ch1)
    angle_list_ch1 = np.angle(cmplx_amp_list_ch1) % (2 * np.pi)
    amp_list_ch3 = abs(cmplx_amp_list_ch3)
    angle_list_ch3 = np.angle(cmplx_amp_list_ch3) % (2 * np.pi)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(np.arange(0, filter_time, 1 / samp_freq), ch1[0:filter_time * samp_freq])
    ax2.plot(np.arange(0, filter_time, 1 / samp_freq), ch3[0:filter_time * samp_freq])
    ax1.set_ylabel("mV")
    ax2.set_ylabel("mV")
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(t0_list, (2 * np.unwrap(angle_list_ch1)))
    ax1.plot(t0_list, np.unwrap(angle_list_ch3))
    ax2.plot(t0_list, amp_list_ch1)
    ax2.plot(t0_list, amp_list_ch3)
    ax1.set_xlabel("us")
    ax2.set_xlabel("us")
    plt.show()

    plt.cla()


# %%
def compute_demod(reset_gage,
                  num_shots_jkam,
                  mask_valid_data_gage,
                  jkam_gage_matchlist,
                  file_prefix_gage,
                  data_path_gage,
                  conversion_factor,
                  num_segments,
                  t0_list,
                  t0_idx_list,
                  filter_time,
                  samp_freq,
                  window,
                  cmplx_amp_array,
                  ch1_pure_vec,
                  ch3_pure_vec,
                  plot_tenth_shot,
                  num_shots_loaded,
                  num_shots_gage,
                  pre_proc_cmplx_amp
                  ):
    if reset_gage:
        print("Processing raw gage files from hard reset...")
        for shot_num in tqdm(range(num_shots_jkam)):
            if mask_valid_data_gage[shot_num]:
                file_name_gage = file_prefix_gage + '_' + str(jkam_gage_matchlist[shot_num]).zfill(5) + '.h5'
                hf = h5py.File(data_path_gage / file_name_gage, 'r')

                for seg_num in range(num_segments):
                    ch1 = np.array(hf[f'CH1_frame{seg_num}']) * conversion_factor  # np.sqrt(n)
                    ch3 = np.array(hf[f'CH3_frame{seg_num}']) * conversion_factor  # np.sqrt(n)

                    cmplx_amp_list_ch1 = t0_list * 0j
                    cmplx_amp_list_ch3 = t0_list * 0j
                    for i, t0_idx in enumerate(t0_idx_list):
                        t1_idx = t0_idx + filter_time * samp_freq
                        if window == 'flattop':
                            w = signal.windows.flattop(t1_idx - t0_idx)
                        elif window == 'square':
                            w = 1
                        elif window == 'hann':
                            w = signal.windows.hann(t1_idx - t0_idx) * 2
                        ch1_demod = np.multiply(ch1[t0_idx:t1_idx] * w, ch1_pure_vec[t0_idx:t1_idx])
                        ch3_demod = np.multiply(ch3[t0_idx:t1_idx] * w, ch3_pure_vec[t0_idx:t1_idx])
                        ch1_demod_sum = np.cumsum(ch1_demod)  # np.concatenate([[0],np.cumsum(ch1_demod)])
                        ch3_demod_sum = np.cumsum(ch3_demod)  # np.concatenate([[0],np.cumsum(ch3_demod)])
                        cmplx_amp_list_ch1[i] = (ch1_demod_sum[-1] - ch1_demod_sum[0]) / (t1_idx - t0_idx)
                        cmplx_amp_list_ch3[i] = (ch3_demod_sum[-1] - ch3_demod_sum[0]) / (t1_idx - t0_idx)

                    cmplx_amp_array[0, shot_num, seg_num] = cmplx_amp_list_ch1
                    cmplx_amp_array[1, shot_num, seg_num] = cmplx_amp_list_ch3

                # if plot_tenth_shot and shot_num == 50:
                #     plot_shots(cmplx_amp_list_ch1,
                #                cmplx_amp_list_ch3,
                #                ch1,
                #                ch3,
                #                t0_list,
                #                samp_freq,
                #                filter_time)
            else:
                cmplx_amp_array[:, shot_num, :, :] = np.array([np.nan])
    else:
        # load in the unpickled file
        cmplx_amp_array = pre_proc_cmplx_amp
        for shot_num in tqdm(range(num_shots_loaded, np.min([num_shots_jkam, num_shots_gage]))):
            if mask_valid_data_gage[shot_num]:
                file_name_gage = file_prefix_gage + '_' + str(jkam_gage_matchlist[shot_num]).zfill(5) + '.h5'
                hf = h5py.File(data_path_gage / file_name_gage, 'r')
                for seg_num in range(num_segments):
                    ch1 = np.array(hf[f'CH1_frame{seg_num}']) * conversion_factor  # np.sqrt(n)
                    ch3 = np.array(hf[f'CH3_frame{seg_num}']) * conversion_factor  # np.sqrt(n)

                    cmplx_amp_list_ch1 = t0_list * 0j
                    cmplx_amp_list_ch3 = t0_list * 0j
                    for i, t0_idx in enumerate(t0_idx_list):
                        t1_idx = t0_idx + filter_time * samp_freq
                        if window == 'flattop':
                            w = signal.windows.flattop(t1_idx - t0_idx)
                        elif window == 'square':
                            w = 1
                        elif window == 'hann':
                            w = signal.windows.hann(t1_idx - t0_idx) * 2
                        ch1_demod = np.multiply(ch1[t0_idx:t1_idx] * w, ch1_pure_vec[t0_idx:t1_idx])
                        ch3_demod = np.multiply(ch3[t0_idx:t1_idx] * w, ch3_pure_vec[t0_idx:t1_idx])
                        ch1_demod_sum = np.cumsum(ch1_demod)  # np.concatenate([[0],np.cumsum(ch1_demod)])
                        ch3_demod_sum = np.cumsum(ch3_demod)  # np.concatenate([[0],np.cumsum(ch3_demod)])
                        cmplx_amp_list_ch1[i] = (ch1_demod_sum[-1] - ch1_demod_sum[0]) / (t1_idx - t0_idx)
                        cmplx_amp_list_ch3[i] = (ch3_demod_sum[-1] - ch3_demod_sum[0]) / (t1_idx - t0_idx)
                    cmplx_amp_array[0, shot_num, seg_num] = cmplx_amp_list_ch1
                    cmplx_amp_array[1, shot_num, seg_num] = cmplx_amp_list_ch3

                # if plot_tenth_shot and shot_num == 10:
                #     plot_shots(cmplx_amp_list_ch1,
                #                cmplx_amp_list_ch3,
                #                ch1,
                #                ch3,
                #                t0_list,
                #                samp_freq,
                #                filter_time)
            else:
                print(f'invalid data at {shot_num:d}')
                cmplx_amp_array[:, shot_num, :, :] = np.array([np.nan])

    return cmplx_amp_array


# %%
def perform_gage_demod(user_constants_dict,
                       arb_constant_dict,
                       gage_constant_dict,
                       jkam_dict,
                       jkam_mask_dict,
                       outer_zoom_factor,
                       num_points_inner,
                       num_points_outer,
                       point_list,
                       num_points,
                       atom_site,
                       num_shots_jkam,
                       date_dir):
    jkam_constants = gen_jkam_constants(user_constants_dict=user_constants_dict,
                                        jkam_dict=jkam_dict,
                                        outer_zoom_factor=outer_zoom_factor,
                                        atom_site=atom_site,
                                        num_points_inner=num_points_inner,
                                        num_points_outer=num_points_outer,
                                        point_list=point_list,
                                        num_points=num_points)
    gage_constants = gen_gage_constants(arb_constant_dict, gage_constant_dict, jkam_constants, date_dir)
    mask_valid_data_gage, jkam_gage_matchlist, gage_index_list = gen_jkam_masks(jkam_mask_dict, jkam_dict,
                                                                                jkam_constants, gage_constants)

    num_shots_gage = gage_constants['num_shots']
    num_shots_loaded = gage_constants['num_shots_loaded']
    file_prefix_gage = gage_constants['file_prefix']
    data_path_gage = gage_constants['data_path']
    conversion_factor = gage_constants['conversion_factor']
    reset_gage = gage_constants['reset_gage']
    window = gage_constants['window']
    num_segments = gage_constants['num_segments']
    plot_tenth_shot = gage_constants['plot_tenth_shot']
    het_freq = gage_constants['het_freq']
    dds_freq = gage_constants['dds_freq']
    samp_freq = gage_constants['samp_freq']
    step_time = gage_constants['step_time']
    filter_time = gage_constants['filter_time']
    run_name = gage_constants['run_name']

    if reset_gage:
        print(f'Hard reset, loading {num_shots_jkam} shots from gage raw data')
        file_name_gage = file_prefix_gage + '_00000.h5'
        hf = h5py.File(data_path_gage / file_name_gage, 'r')
        chlen = len(np.array(hf.get('CH1_frame0')))
        t_vec = np.arange(chlen) * 1 / samp_freq
        ch1_pure_vec = np.exp(-1j * 2 * np.pi * dds_freq * t_vec)
        ch3_pure_vec = np.exp(-1j * 2 * np.pi * het_freq * t_vec)

        t0_list = np.arange(0, chlen / samp_freq - filter_time + step_time, step_time)
        timebin_array = np.empty((len(t0_list), 2), dtype=float)
        timebin_array[:, 0] = t0_list
        timebin_array[:, 1] = t0_list + filter_time
        cmplx_amp_array = np.empty((2, num_shots_gage, num_segments, len(t0_list)), dtype=np.cdouble)
        t0_idx_list = np.arange(0, chlen - filter_time * samp_freq + 1, step_time * samp_freq)

        cmplx_amp_array = compute_demod(reset_gage,
                                        num_shots_jkam,
                                        mask_valid_data_gage,
                                        jkam_gage_matchlist,
                                        file_prefix_gage,
                                        data_path_gage,
                                        conversion_factor,
                                        num_segments,
                                        t0_list,
                                        t0_idx_list,
                                        filter_time,
                                        samp_freq,
                                        window,
                                        cmplx_amp_array,
                                        ch1_pure_vec,
                                        ch3_pure_vec,
                                        plot_tenth_shot,
                                        num_shots_loaded,
                                        num_shots_gage,
                                        cmplx_amp_array)
        outpath = jkam_dict['working_path'] / 'analysis'
        outpath = str(outpath)
        # use os to make this folder
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        with open(outpath + '\\' + f'{run_name}_{window}_gage_cmplx_amp_{filter_time}_{step_time}.pkl', 'wb') as f1:
            pickle.dump(cmplx_amp_array, f1)
        with open(outpath + '\\' + f'{run_name}_{window}_gage_timebin_{filter_time}_{step_time}.pkl', 'wb') as f3:
            pickle.dump(timebin_array, f3)
        print('done')

    else:
        outpath = jkam_dict['working_path'] / 'analysis'
        outpath = str(outpath)
        # use os to make this folder
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        print(f'loading {num_shots_loaded} shots from gage pickle files')
        try:
            cmplx_amp_array_old = np.load(
                outpath + '\\' + f'{run_name}_{window}_gage_cmplx_amp_{filter_time}_{step_time}.pkl',
                allow_pickle=True)
            timebin_array = np.load(outpath + '\\' + f'{run_name}_{window}_gage_timebin_{filter_time}_{step_time}.pkl',
                                    allow_pickle=True)
        except Exception as e:
            print('first time run. Loading pkl failed due to: ', e)

        if (num_shots_jkam > num_shots_loaded):
            print(f'loading {num_shots_loaded} to {np.min([num_shots_jkam, num_shots_gage])} shots from gage raw data')
            file_name_gage = file_prefix_gage + '_00000.h5'
            hf = h5py.File(data_path_gage / file_name_gage, 'r')
            chlen = len(np.array(hf.get('CH1_frame0')))
            t_vec = np.arange(chlen) * 1 / samp_freq
            ch1_pure_vec = np.exp(-1j * 2 * np.pi * dds_freq * t_vec)
            ch3_pure_vec = np.exp(-1j * 2 * np.pi * het_freq * t_vec)
            t0_list = np.arange(0, chlen / samp_freq - filter_time + step_time, step_time)
            timebin_array = np.empty((len(t0_list), 2), dtype=float)
            timebin_array[:, 0] = t0_list
            timebin_array[:, 1] = t0_list + filter_time
            cmplx_amp_array = np.empty((2, num_shots_gage, num_segments, len(t0_list)), dtype=np.cdouble)

            if num_shots_loaded > 0:
                cmplx_amp_array[:, 0:num_shots_loaded, :] = cmplx_amp_array_old
            t0_idx_list = np.arange(0, chlen - filter_time * samp_freq + 1, step_time * samp_freq)

            cmplx_amp_array = compute_demod(reset_gage,
                                            num_shots_jkam,
                                            mask_valid_data_gage,
                                            jkam_gage_matchlist,
                                            file_prefix_gage,
                                            data_path_gage,
                                            conversion_factor,
                                            num_segments,
                                            t0_list,
                                            t0_idx_list,
                                            filter_time,
                                            samp_freq,
                                            window,
                                            cmplx_amp_array,
                                            ch1_pure_vec,
                                            ch3_pure_vec,
                                            plot_tenth_shot,
                                            num_shots_loaded,
                                            num_shots_gage,
                                            cmplx_amp_array)

            with open(outpath + '/' + f'{run_name}_{window}_gage_cmplx_amp_{filter_time}_{step_time}.pkl', 'wb') as f1:
                pickle.dump(cmplx_amp_array, f1)
            with open(outpath + '/' + f'{run_name}_{window}_gage_timebin_{filter_time}_{step_time}.pkl', 'wb') as f3:
                pickle.dump(timebin_array, f3)
            print('done')
        else:
            cmplx_amp_array = cmplx_amp_array_old
    print(f"number of {num_shots_jkam} shots loaded")


# this will need to run in the background as the other programs run, so we won't have
# the opportunity to use the analysis notebook scheme of reading in a bunch of parameters

# ENTER YOUR DESIRED PRE-PROCESSING PARAMETERS HERE PRIOR TO THE RUN BEGINNING
# TODO: READ NUM SEGMENTS AUTOMATICALLY FROM GAGE FILE
# TODO: BUG: IF THE MATRIX GETS TOO BIG (AKA BIGGER THAN 4GB IT FAILS)
# TODO: BUG: IF THERE ARE MORE JKAM SHOTS THAN GAGE SHOTS, THAN THE FILE PROCESSING FAILS
def populate_dicts(root_dir, run_name, date_dir):
    # the root dir we will get from photon timer "full_dir" parameter
    #############################################################
    # USER INPUTS
    working_path = Path(root_dir)
    user_dict = {'run_name': run_name,
                 'override_num_shots': False,
                 'reset_hard': False,
                 'num_shots_manual': 286,
                 'num_frames': 3,
                 'point_name_inner': 'delta_pc (MHz)',
                 'point_name_outer': 'pump power (uW)',
                 'point_list_outer': ControlV_Power(np.array([0.8, 0.9, 1, 1.1, 1.2, 1.3]), 150),
                 'point_list_inner': np.array([-2.45, -2.3, -2.15, -2, -1.85, -1.7, -1.55]),
                 'point_parallel': False,
                 'tweezer_freq_list': 88 + 0.8 * 0 + 0.8 * np.arange(40)}

    jkam_dict = {'datastream_name': 'High NA Imaging',
                 'working_path': working_path,
                 'file_prefix': 'jkam_capture',
                 'num_shot_start': 0}
    gage_constant_dict = {'reset_gage': False,
                          'window': 'hann',
                          'num_segments': 22,
                          'plot_tenth_shot': True,
                          'het_freq': 20.000446,  # MHz
                          'dds_freq': 20.000446 / 2,
                          'samp_freq': 200,  # MHz
                          'step_time': 1,  # us
                          'filter_time': 5,  # us
                          'datastream_name': 'gage',
                          'working_path': working_path,
                          'run_name': run_name,
                          'file_prefix': 'gage_shot',
                          'file_suffix': '.h5'}

    data_path = jkam_dict['working_path'] / 'data' / user_dict['run_name'] / jkam_dict['datastream_name']
    path, dirs, files = next(os.walk(data_path))
    num_shots = len(files)

    if len(user_dict['point_list_inner']) == 1:
        outer_zoom_factor = 1
    else:
        outer_zoom_factor = 10

    atom_site = []
    for i in range(user_dict['num_frames']):
        atom_site.append(np.arange(len(user_dict['tweezer_freq_list'])))

    num_points_inner = len(user_dict['point_list_inner'])
    num_points_outer = len(user_dict['point_list_outer'])
    if num_points_inner == 1:
        point_list = np.array(user_dict['point_list_outer'])
    elif user_dict['point_parallel'] == False:
        point_list = (outer_zoom_factor * np.outer(user_dict['point_list_outer'],
                                                   np.ones(len(user_dict['point_list_inner'])))
                      + np.outer(np.ones(len(user_dict['point_list_outer'])),
                                 user_dict['point_list_inner'])).flatten()
    elif user_dict['point_parallel'] == True:
        point_list = np.arange(num_points_outer * num_points_inner)
        plt.plot(point_list)
    num_points = len(point_list)

    addtl_consts = {'outer_zoom_factor': outer_zoom_factor,
                    'num_points_inner': num_points_inner,
                    'num_points_outer': num_points_outer,
                    'point_list': point_list,
                    'num_points': num_points,
                    'atom_site': atom_site,
                    'num_shots': num_shots}

    jkam_const = gen_jkam_constants(user_dict,
                                    jkam_dict,
                                    outer_zoom_factor,
                                    atom_site,
                                    num_points_inner,
                                    num_points_outer,
                                    point_list,
                                    num_points)

    jkam_mask_dict = gen_jkam_mask_info(jkam_dict, jkam_const)
    gen_constant_dict = {'voltage_conversion': 1000 / 32768,
                         'kappa': 2 * np.pi * 1.1,
                         'LO_power': 314,
                         'PHOTON_ENERGY': 2.55e-19}

    gage_constant_dict = gen_gage_constants(gen_constant_dict, gage_constant_dict, jkam_const, date_dir)

    return gage_constant_dict, gen_constant_dict, user_dict, jkam_mask_dict, jkam_dict, addtl_consts


def naive_impl(root_dir, run_name, date_dir):
    # nothing fancy, just keep stacking more and more pickl files after waiting 15 seconds
    gage_constant_dict, gen_constant_dict, user_dict, jkam_mask_dict, jkam_dict, addtl_consts = populate_dicts(root_dir,
                                                                                                               run_name,
                                                                                                               date_dir)
    datastream_name_gage = gage_constant_dict['datastream_name']
    working_path_gage = gage_constant_dict['working_path']
    run_name = gage_constant_dict['run_name']
    gage_path = working_path_gage / 'data' / run_name / datastream_name_gage
    curr_length = len(os.listdir(gage_path))
    first_run = True

    while True:
        # TODO: MORE ELEGANTLY FIX BUG WHERE THE NUMBER OF SHOTS IS NOT UPDATED PROPERLY
        new_length = len(os.listdir(gage_path))
        gage_constant_dict, gen_constant_dict, user_dict, jkam_mask_dict, jkam_dict, addtl_consts = populate_dicts(
            root_dir, run_name, date_dir)
        print("Current length of gage path: ", curr_length)
        if first_run or new_length > curr_length:
            try:
                perform_gage_demod(user_constants_dict=user_dict,
                                   arb_constant_dict=gen_constant_dict,
                                   gage_constant_dict=gage_constant_dict,
                                   jkam_dict=jkam_dict,
                                   jkam_mask_dict=jkam_mask_dict,
                                   outer_zoom_factor=addtl_consts['outer_zoom_factor'],
                                   num_points_inner=addtl_consts['num_points_inner'],
                                   num_points_outer=addtl_consts['num_points_outer'],
                                   point_list=addtl_consts['point_list'],
                                   num_points=addtl_consts['num_points'],
                                   atom_site=addtl_consts['atom_site'],
                                   num_shots_jkam=addtl_consts['num_shots'],
                                   date_dir = date_dir)

                print("files processed again. current file number is : ", new_length)
                time.sleep(2)
                first_run = False
                curr_length = new_length
            except Exception as e:
                print("processing failed. The following error occurred: ", e)

            time.sleep(2)
            first_run = False
            curr_length = new_length
            time.sleep(5)
        else:
            print("No new files to process.")
            time.sleep(5)
            continue


if __name__ == '__main__':
    dir_data = ('X:/expdata-e6/data')
    date_dir = datetime.datetime.now().strftime("%Y/%m/%d/")
    # USE THIS TO SET THE MANUAL DATE IF YOU WANT TO SELECT A SPECIFIC DAY TO PROCESS
    manual_date = '2024/07/25'
    date_dir = manual_date
    ################ CHANGE THE RUN NUMBER AS YOU GO
    SeqRunName = 'run1'
    full_dir = dir_data + '/' + date_dir
    ################ UNCOMMENT THIS TO USE THE MANUAL DATE
    # full_dir = dir_data + '/' + manual_date
    naive_impl(full_dir, SeqRunName, date_dir)

    # TODO: DAY NIGHT CYCLE ISSUE: IF THE RUN GOES OVERNIGHT, THE PROGRAM
    # WILL CHANGE TO LOOKING FOR PICKLE FILES IN THE NEW DAY, FOR WHICH
    # A DATA FOLDER DOES NOT EXIST!!! THIS IS HOW WE NEED TO RESOLVE IT:
    #
    # STORE A STARTING DATETIME
    # IF DATETIME.NOW IS DIFFERENT FROM THE STARTING DATETIME, CONTINUE LOOKING
    # FOR FILES IN THE STARTING DATETIME
    # OR WE COULD JUST FIX THE STARTING DATETIME TO THE INITIAL VALUE
