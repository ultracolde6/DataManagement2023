#run this the first time to spot any file offset glitches

datastream_name='High NA Imaging'
working_path = Path.cwd().parent
data_path = run_name + '/' + datastream_name
file_prefix='jkam_capture'
path, dirs, files = next(os.walk( data_path ))

num_shots_jkam = len(files)
num_shots = num_shots_jkam
num_loops = num_shots//num_points
num_tweezers = len(tweezer_freq_list)
num_points = len(point_list)
num_loops = num_shots//num_points + bool(num_shots%num_points)
print(f'num_points={num_points}, num_loops={num_loops}')
print(f'num_shots={num_shots}')

if override_num_shots:
    num_shots = num_shots_manual

# path, dirs, files = next(os.walk(data_path))
# num_shots_fpga = len(files)
# # num_shots = num_shots_fpga
# if override_num_shots:
#     num_shots = num_shots_manual
jkam_creation_time_array =  np.zeros(num_shots)


for shot_num in range(num_shots):    
    file_name = file_prefix+'_'+str(shot_num).zfill(5)+'.h5'
    jkam_creation_time_array[shot_num] = os.path.getctime(data_path + '/' + file_name)
    if shot_num%1000 ==0:
        print(shot_num)
        
avg_time_gap = (jkam_creation_time_array[-1]-jkam_creation_time_array[0])/(num_shots-1)
# fpga_creation_time_gap = fpga_creation_time_array[1:]-fpga_creation_time_array[:-1]
        
datastream_name='PhotonTimer'
working_path = Path.cwd().parent
data_path_fpga = run_name + '/' + datastream_name
fpga_file_prefix='PTPhotonTimer'
fpga_creation_time_array =  np.zeros(num_shots)

opal_path, opal_dirs, opal_files = next(os.walk( data_path_fpga ))

num_shots_fpga=len(opal_files)

for shot_num in range(num_shots):
    if shot_num<num_shots_fpga:
        file_name = fpga_file_prefix+'_'+str(shot_num).zfill(5)+'.bin'
        fpga_creation_time_array[shot_num] = os.path.getctime(data_path_fpga + '/' + file_name)
        if shot_num%1000 ==0:
            print(shot_num)

#Check data matching
mask_valid_data=np.zeros(len(jkam_creation_time_array))>1
jkam_fpga_matchlist=np.zeros(len(jkam_creation_time_array),dtype='int')-1
fpga_index_list=np.arange(len(fpga_creation_time_array))


for shot_num in range(num_shots):
    time_temp=jkam_creation_time_array[shot_num]
    space_correct=True
    if (True): space_correct=False
    if (True):
        if (True): space_correct=False
            
    if (True):
        mask_valid_data[shot_num]=True
        jkam_fpga_matchlist[shot_num]=fpga_index_list[np.argmin(np.abs(fpga_creation_time_array-time_temp))]
    else:
        print(f'error at {shot_num:d}')
    if shot_num%1000 ==0:
            print(shot_num)