"""
Script to calculate Iv parameters of IV curves
-Guillermo Farias Basulto
-last update: 23.08.2023
This script should:
    Process all files, in a folder selected by the user, containing the strings:
        "ID #-#-#-# Cell # hell.dat"
        "ID #-#-#-# Cell # dunkel.dat"
        "ID #-#-#-# Parameter.dat"

    From each file the BAdngap (Eg) is extracted using different methods:
      1 From the rule of thumb: 40 percent EQE in the far infrared slope
      2 Minimum first derivative (i.e. dEQE/dWavelength)
      3 Interpolation method: interpolating the far infrared slope to redefine the values in the slope
      4 Linear regression using the intercept at X
      5 Smoothening of the EQE (moving average) and then using the derivative method
    Show or save the plots and values

"""
import pandas as pd
import os
import re
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askdirectory

import Data_functions

global save_results_to_txt,id_string,cell_number, save_plots,plot_all_IV_in_One,plot_dark_IVs,plot_light_IVs
global compare_variation,compare_IDs
'''Select variables for saving and showing plots'''
'''----RELEVANT FOR USER!!-----'''
search_for_directory = 0
versuchsplan ="VP1111"
plot_light_IVs =0
plot_dark_IVs =0
plot_all_IV_in_One =0
save_plots = 0  #saves the plots but does not show them
save_results_to_txt = 1  # Saves all results in a txt file
boxplots=1
compare_samples = 1  ##To compare samples , This adds the name of the variation to specific IDs
if compare_samples== 1:
    group_variations=0
    compare_variation =['var1', 'var2']   #       compare_variation =['var1', 'var2']   #  'ID 2-203-5-2'--> var1
    compare_IDs =['ID 2-203-5-2', 'ID 2-203-5-3']   #    compare_IDs =['ID 2-203-5-2', 'ID 2-203-5-3']
#####

'''----NOT RELEVANT FOR USER!!-----'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if search_for_directory == 1:
        working_directory = Data_functions.get_directory()
    else:
        working_directory = os.getcwd()
    #Takes all files in the directory and makes a list of dataframes for all files
    dataframes_hell, dataframes_dunkel, dataframes_parameter,dataframe_statistics = Data_functions.IV_file(working_directory)
    #results = pd.DataFrame(columns=[])  ### for final results

    graph_title_all = []  ##List for the leayend of each EQE for plot
    IV_all_x = []  ## List to save all nm data for plot
    IV_all_y = []  ## List to save all IV data for plot

    #FOR LIGHT IV processing------------------------------------------------------------------------------
    #-------------------
    for dat_file, df in dataframes_hell.items():  # Print the DataFrames.
        # try:
        print(f"File name: {dat_file}")
        id_string, cell_number = df['Sample'][1] , df['cell#'][1]
                    #########  light IV ###############
        iv_data = df.iloc[:, 0:2]  # get only the first two col
        ##Maybe later make a class with methods to to extract IV parameters from each IV

        '''##Plot IVs----------------------------'''
        if plot_light_IVs == 1:
            print("plot IVs on")
            graph_title = 'IV_' + id_string + '_#' + cell_number
            Data_functions.plot_iv(graph_title, "Voltage [V]", "Current [mA/cm²]", df["voltage in V"], df["current density in mA/cm²"])
            if save_plots == 1:
                plt.savefig('plt_' + graph_title + '.png')
            else:
                plt.show()
            if plot_all_IV_in_One == 1:
                graph_title_all.append(graph_title)  ##appends all nm lists into the list for plotting
                IV_all_x.append(df["voltage in V"].tolist())  ##appends all nm lists into the list for plotting
                IV_all_y.append(df["current density in mA/cm²"].to_list())  ##appends all EQEs lists into the list for plotting
                #print(graph_title_all)
        ### Saves a concatenated file for data

    if plot_light_IVs or save_plots  == 1:
        Data_functions.plot_iv(graph_title_all, "Voltage [V]", "Current [mA/cm²]", IV_all_x, IV_all_y)
    if save_plots == 1:
        plt.savefig('plt_all_' + versuchsplan + '.png')
    else:
        plt.show()

    '''# FOR DARK IV processing-----------------------------------------------------------------------'''
    #-------------------
    for dat_file, df in dataframes_dunkel.items():  # Print the DataFrames.
        # try:
        print(f"File name: {dat_file}")
        id_string, cell_number = df['Sample'][1], df['cell#'][1]
        #########  light IV ###############
        iv_data = df.iloc[:, 0:2]  # get only the first two col
        ##Maybe later make a class with methods to to extract IV parameters from each IV

        ##Plot IVs
        if plot_light_IVs == 1:
            print("plot IVs on")
            graph_title = 'IV_' + id_string + '_#' + cell_number
            Data_functions.plot_iv(graph_title, "Voltage [V]", "Current [mA/cm²]", df["voltage in V"],
                                   df["current density in mA/cm²"])
            if save_plots == 1:
                plt.savefig('plt_' + graph_title + '.png')
            else:
                plt.show()
            if plot_all_IV_in_One == 1:
                graph_title_all.append(graph_title)  ##appends all nm lists into the list for plotting
                IV_all_x.append(df["voltage in V"].tolist())  ##appends all nm lists into the list for plotting
                IV_all_y.append(
                    df["current density in mA/cm²"].to_list())  ##appends all EQEs lists into the list for plotting
                # print(graph_title_all)
        ### Saves a concatenated file for data
    if plot_light_IVs or save_plots  == 1:
        Data_functions.plot_iv(graph_title_all, "Voltage [V]", "Current [mA/cm²]", IV_all_x, IV_all_y)
    if save_plots == 1:
        plt.savefig('plt_all_' + versuchsplan + '.png')
    else:
        plt.show()

    '''# FOR Statistics and parameter value saving------------------------------------------------'''
    # -------------------
    #result_parameters = []
    print('---------------------PARAMETERS-------------------------')

    if save_results_to_txt == 1:
        results_statistics = pd.DataFrame()
        results_parameters = pd.DataFrame()
        for dat_file, df in dataframes_parameter.items():  # Print the DataFrames.
            results_parameters = pd.concat([results_parameters, df], axis=0)  ##Append all data in one dataframe
        for dat_file, df in dataframe_statistics.items():  # Print the DataFrames.
            results_statistics = pd.concat([results_statistics, df], axis=0)  ##Append all data in one dataframe
        results_parameters.insert(0, 'VP', versuchsplan)  ##insert array 'VP' in the dataframe df_data, in column 0;
        results_statistics.insert(0, 'VP', versuchsplan)  ##insert array 'VP' in the dataframe df_data, in column 0;
        if compare_samples ==1:
            compare_var_n_IDs = []
            results_parameters= Data_functions.include_variations(results_parameters, compare_variation, compare_IDs)  ##generates a df adding the variation to each ID
            results_statistics= Data_functions.include_variations(results_parameters, compare_variation, compare_IDs)  ##generates a df adding the variation to each ID
            if group_variations == 1:  # Should the samples be compared regardles of sample ID
                df = df.sort_values(by=['Variations'], ascending=True)
        print('result_parameters----end ')
        print(results_parameters.to_string())
        print('result_stats----end')
        print(results_statistics.to_string())

        txt_results = open(working_directory + "/"+versuchsplan+"_all_parameter_results.txt", "w+")
        txt_results.write(results_parameters.to_string())
        txt_results.close()
        txt_results = open(working_directory + "/"+versuchsplan+"_stats_results.txt", "w+")
        txt_results.write(results_statistics.to_string())
        txt_results.close()

    if boxplots==1:
        all_sample_IDs = results_parameters['Sample'].drop_duplicates()  # gets only non repeated values of IDs
        #all_sample_IDs = np.array(all_sample_IDs)
        all_sample_IDs = list(all_sample_IDs)

        print(all_sample_IDs)
        parameters_to_plot= ["Jsc in mA/cm²", "Voc in mV", "FF in %", "Roc in Ohm*cm²", "Rsc in Ohm*cm²", "eta in %"]
        Data_functions.plot_all_boxplots_per_sample(results_parameters,parameters_to_plot, all_sample_IDs, versuchsplan) #data, parameter,ID

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
