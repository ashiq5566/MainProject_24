import os
import numpy as np
import pandas as pd

    
def read_files():
    mypath = "C:/Users/User/OneDrive - Nanyang Technological University/LTA/Experiment 1/SensorKineticsCharts"
    mypath = os.path.normpath(mypath)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))] #contains all the files in the directory
    journey_names = []

    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext == ".csv":
            journey_name = os.path.splitext(file)[0].lower().split('_')[0]
            if journey_name not in journey_names:
                journey_names.append(journey_name)
    
    accm_df_list = []
    gyrm_df_list = []
    lacm_df_list = []
    
    for journey_name in journey_names:
        #for accelerometer data
        accm_df = pd.read_csv(os.path.join(mypath, journey_name+ "_accm.csv"),skiprows=[1], 
                            dtype={'time':np.float,
                                   'X_value':np.float, 
                                   'Y_value':np.float,
                                   'Z_value':np.float})
        accm_df['JourneyID'] = journey_name
        if "bus" in journey_name: accm_df['Mode'] = 'Bus'
        if 'mrt' in journey_name: accm_df['Mode'] = 'MRT'
        if 'walk' in journey_name: accm_df['Mode'] = 'Walk'
        if 'idle' in journey_name: accm_df['Mode'] = 'Idle'
        if 'static' in journey_name: accm_df['Mode'] = 'Static'
        accm_df_list.append(accm_df)
        
        #for gyrometer data
        gyrm_df = pd.read_csv(os.path.join(mypath, journey_name+ "_gyrm.csv"),skiprows=[1], 
                            dtype={'time':np.float,
                                   'X_value':np.float, 
                                   'Y_value':np.float,
                                   'Z_value':np.float})
        gyrm_df['JourneyID'] = journey_name
        if "bus" in journey_name: gyrm_df['Mode'] = 'Bus'
        if 'mrt' in journey_name: gyrm_df['Mode'] = 'MRT'
        if 'walk' in journey_name: gyrm_df['Mode'] = 'Walk'
        if 'idle' in journey_name: gyrm_df['Mode'] = 'Idle'
        if 'static' in journey_name: gyrm_df['Mode'] = 'Static'
        gyrm_df_list.append(gyrm_df)
        
        #for linear acceleration data
        gyrm_df = pd.read_csv(os.path.join(mypath, journey_name+ "_lacm.csv"),skiprows=[1], 
                            dtype={'time':np.float,
                                   'X_value':np.float, 
                                   'Y_value':np.float,
                                   'Z_value':np.float})
        gyrm_df['JourneyID'] = journey_name
        if "bus" in journey_name: gyrm_df['Mode'] = 'Bus'
        if 'mrt' in journey_name: gyrm_df['Mode'] = 'MRT'
        if 'walk' in journey_name: gyrm_df['Mode'] = 'Walk'
        if 'idle' in journey_name: gyrm_df['Mode'] = 'Idle'
        if 'static' in journey_name: gyrm_df['Mode'] = 'Static'
        lacm_df_list.append(gyrm_df)
        
        
    all_accm_df = pd.DataFrame(columns = ['time', 'X_value', 'Y_value', 'Z_value', 'JourneyID', 'Mode'])
    all_accm_df = pd.concat(accm_df_list)
    all_accm_df.rename(columns={"X_value": "acc_X", "Y_value": "acc_Y", "Z_value": "acc_Z"}, inplace=True)
    all_accm_df.reset_index(inplace=True,drop=True)
    
    all_gyrm_df = pd.DataFrame(columns = ['time', 'X_value', 'Y_value', 'Z_value', 'JourneyID', 'Mode'])
    all_gyrm_df = pd.concat(gyrm_df_list)
    all_gyrm_df.rename(columns={"X_value": "gyr_X", "Y_value": "gyr_Y", "Z_value": "gyr_Z"}, inplace=True)
    all_gyrm_df.reset_index(inplace=True,drop=True)
    
    all_lacm_df = pd.DataFrame(columns = ['time', 'X_value', 'Y_value', 'Z_value', 'JourneyID', 'Mode'])
    all_lacm_df = pd.concat(gyrm_df_list)
    all_lacm_df.rename(columns={"X_value": "gyr_X", "Y_value": "gyr_Y", "Z_value": "gyr_Z"}, inplace=True)
    all_lacm_df.reset_index(inplace=True,drop=True)

          
    return all_accm_df, all_gyrm_df, all_lacm_df

def moving_window(accm_df, gyrm_df, lacm_df, time_per_journey = 2, overlap = 50, original_sampling_rate = 400, target_sampling_rate=400):
    #Making calculations
    downsampling_rate =  target_sampling_rate / original_sampling_rate
    data_points_skipped =  int(1 / downsampling_rate)
    time_increment = time_per_journey * overlap // 100 
    threshold = int(target_sampling_rate * time_per_journey * 0.875)
    #0.875 is set as a constant to ensure so that for a 2s per journey slicing window, 
    #data sets of at least (2s * 0.875 = 1.75s) are added as well instead of discarded.
    
    #Initializing values
    accm_slicers=[]
    gyrm_slicers=[]
    lacm_slicers=[]
    accm_max_index = 0
    gyrm_max_index = 0
    lacm_max_index = 0
    
    #Downsampling accelerometer data
    accm_df_copy = accm_df[0: len(accm_df): data_points_skipped]
    
    #Downsampling gyrometer data
    gyrm_df_copy  = gyrm_df[0: len(gyrm_df): data_points_skipped]
    
    #Downsampling linear acceleration meter data
    lacm_df_copy  = lacm_df #[0: len(gyrm_df): data_points_skipped]
    
    #Iterating through all unique journeys
    unique_journey = accm_df['JourneyID'].unique()
    
    for journey in unique_journey:
        accm_timestamp = accm_df_copy.loc[accm_df['JourneyID']==journey,:]['time']
        gyrm_timestamp  = gyrm_df_copy.loc[gyrm_df['JourneyID']==journey,:]['time']
        lacm_timestamp  = lacm_df_copy.loc[gyrm_df['JourneyID']==journey,:]['time']

        time_max = gyrm_timestamp.max() #or using accm_timestamp.max() would work too
        
        gyrm_timestamp = gyrm_timestamp.values.ravel() #converting time from pd.Series to np array
        accm_timestamp = accm_timestamp.values.ravel() #converting time from pd.Series to np array
        lacm_timestamp = lacm_timestamp.values.ravel() #converting time from pd.Series to np array
        
        t = 0
        
        if time_max > 10: #only use samples that are more than 12 seconds.
            #Implementing sliding window for gyrometer and accelerometer simultaneously.
            while t + 3 * time_per_journey < time_max: #the multiplier 3 is to remove the last 6s of each data.
                gyrm_idx_start = np.searchsorted(gyrm_timestamp, t) #finds i where timestamp[i-1] < t <= timestamp[i]
                gyrm_idx_end = np.searchsorted(gyrm_timestamp, t + time_per_journey) 
                
                accm_idx_start = np.searchsorted(accm_timestamp, t) #finds i where timestamp[i-1] < t <= timestamp[i]
                accm_idx_end = np.searchsorted(accm_timestamp, t + time_per_journey) 
                
                lacm_idx_start = np.searchsorted(lacm_timestamp, t) #finds i where timestamp[i-1] < t <= timestamp[i]
                lacm_idx_end = np.searchsorted(lacm_timestamp, t + time_per_journey) 
                t+= time_increment 
                
                if all([(gyrm_idx_end - gyrm_idx_start) >= threshold, (accm_idx_end - accm_idx_start) >= threshold, (lacm_idx_end - lacm_idx_start) >= threshold ]): 
                    #If there are more than certain number of data points for both gyrm and accm, then
                    #we append the 2s sliding window.
                    accm_idx_list = [accm_idx_start+ accm_max_index, accm_idx_end+ accm_max_index]
                    gyrm_idx_list = [gyrm_idx_start+ gyrm_max_index, gyrm_idx_end+ gyrm_max_index]
                    lacm_idx_list = [lacm_idx_start+ lacm_max_index, lacm_idx_end+ lacm_max_index]
                    accm_slicers.append(accm_idx_list)
                    gyrm_slicers.append(gyrm_idx_list)
                    lacm_slicers.append(lacm_idx_list)
                else:
                    pass 
            
        accm_max_index += (accm_df_copy['JourneyID']==journey).sum()
        gyrm_max_index += (gyrm_df_copy['JourneyID']==journey).sum()
        lacm_max_index += (lacm_df_copy['JourneyID']==journey).sum()
        
    return accm_slicers, accm_df_copy, gyrm_slicers, gyrm_df_copy, lacm_slicers, lacm_df_copy

def get_df_window(df, slicer,i):
    """This function takes in the df and slicer (the output of moving_window function)
    to access the ith moving window."""
    slicer = gyrm_slicers[i]
    idx = np.arange(slicer[0], slicer[1])
    return df.iloc[idx,:]

gyrm_df_list = []
for i in range(len(gyrm_slicers)):
    slicer = gyrm_slicers[i]
    idx = np.arange(slicer[0], slicer[1])
    gyrm_df_list.append(gyrm_df_copy.iloc[idx,:])

accm_df_list = []
for i in range(len(accm_slicers)):
    slicer = accm_slicers[i]
    idx = np.arange(slicer[0], slicer[1])
    accm_df_list.append(accm_df_copy.iloc[idx,:])
    
lacm_df_list = []
for i in range(len(lacm_slicers)):
    slicer = lacm_slicers[i]
    idx = np.arange(slicer[0], slicer[1])
    lacm_df_list.append(lacm_df_copy.iloc[idx,:])
    
# #Creating a list to match Overlapped Journey Index with Mode.

car_overlapped_journeys = []
idle_overlapped_journeys = []
bus_overlapped_journeys = []
mrt_overlapped_journeys = []
walk_overlapped_journeys = []
static_overlapped_journeys = []

for i in range(len(accm_df_list)):
    if accm_df_list[i].Mode.iloc[0] == 'Car':
        car_overlapped_journeys.append(i)
    elif accm_df_list[i].Mode.iloc[0] == 'Idle':
        idle_overlapped_journeys.append(i)
    elif accm_df_list[i].Mode.iloc[0] == 'Bus':
        bus_overlapped_journeys.append(i)
    elif accm_df_list[i].Mode.iloc[0] == 'MRT': 
        mrt_overlapped_journeys.append(i)
    elif accm_df_list[i].Mode.iloc[0] == 'Walk': 
        walk_overlapped_journeys.append(i)
    elif accm_df_list[i].Mode.iloc[0] == 'Static':
        static_overlapped_journeys.append(i)

print('The number of bus journeys are %s' %len(bus_overlapped_journeys) )
print('The number of car journeys are %s' %len(car_overlapped_journeys) )
print('The number of idle journeys are %s' %len(idle_overlapped_journeys) )
print('The number of MRT journeys are %s' %len(mrt_overlapped_journeys) )
print('The number of walk journeys are %s' %len(walk_overlapped_journeys) )
print('The number of static journeys are %s' %len(static_overlapped_journeys) )
