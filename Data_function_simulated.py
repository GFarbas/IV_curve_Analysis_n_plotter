import pandas as pd
import os
import re
import numpy as np
from matplotlib import cbook
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askdirectory

def IV_file_simu(working_directory):  ##adjust the info in the file to be handeld in a dataframe
    '''Use only for IVs in one folder which are generated in the same simulation run'''
    simulation_files= []
    dataframes_simu = {}    # Create a list of .dat files in the current working directory.
    dataframes_parameter_simu= {}    # Create a dictionary to store the DataFrames.
    cwd = os.getcwd()

    for file in os.listdir(working_directory):  #store all legible files in a dataframe
        if file.endswith(".txt") and 'IV-T' in file:
            simulation_files.append(file)  ##store all IV files
            ##columns in file [VOLTAGE,CURRENT,CURRENTDIODE,SHUNTCURRENT,POWER]
        elif file.endswith(".xlsx") and 'SimulatedParameters' in file:
            dataframes_parameter_simu = pd.read_excel(file)  #save as dataframe the parameters used for simulation
            #columns in XLS [Date	ID-No	Jsc[A]	Voc[V]	FF	P_out[W]	Eta[%]	Suns	Rs	Rp	Temp[K]	Ideality	J0	J00	Eg	Vmpp	Jmpp	FileNameIV]
        else:
            print("Not possible with file: ", file)

        for dat_file in simulation_files:  # For each .txt file, create a DataFrame and store it in the dictionary.
            print("Simulated IV file: ", dat_file)
            filepath = os.path.join(cwd, working_directory, dat_file)
            #df2 = file_separation_IV_simu(dat_file)  ## prepares data to get only the IV
            df2 = pd.read_csv(filepath, header=0)
            df2 = pd.DataFrame(df2)  # Create a DataFrame from the list of lists.
            #df2["file_name"] = dat_file  # Save the file´s name in the DataFrame.
            #df2['measurement_type'] = 'light'
            dataframes_simu[dat_file] = df2  # Add the DataFrame to the dictionary.
        ## fitting of IV to be included here
        #for dat_file in dataframes_parameter_simu:  # For each .xls file, create a DataFrame and store it in the dictionary.
        #    print("Parameter file: ", dat_file)
        #    df3, df4 = file_separation_IV_parameters(dat_file)
        #    dataframes_parameter[dat_file] =df3
        #    statistics_IV[dat_file]  =df4

    else:
        print("Not possible with file: ", dat_file)

    return dataframes_simu, dataframes_parameter_simu

