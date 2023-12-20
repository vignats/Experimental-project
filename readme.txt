patients: MEPPAVOP 41-50, 61 2022-2023 
file generated from main_pa_estimation_2023.m

date: ms are missing
delay_s: time since the first date, unit = s. 
raw-pat: all pat values, outliers should be removed, unit = s. 
filtered_pat: PAT, after a selection step, and time smooth on 20s window ; to be compare with raw pat. unit = s. 
mask: validation of PAT values based on the amplitude/noise/shape of the PPG signal; not perfect, could be improved. logical. 
BP systolic / mean / diastolic: unit = mmHg. 
