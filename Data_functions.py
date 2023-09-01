"""
Useful function for handling files and plot graphs
-Guillermo Farias Basulto
-last update: 23.08.2023
This script has the following functions:
    find_string_in_list                 # Finds a string on a list
    get_values_around_index             # Gets a define number of values around a given index
    interpolate_values                  # Interpolates x and y values
    get_better_range                    #
    get_slope                           # Makes linear regression and gives back the solpe and intercept
    is_monotonic_increasing             # Checks if a data series are monotonically increasing [1,2,3,4]
    get_minimum_derivative_index        # Finds the minimum derivative of a curve and gives the location index
    file_separation_eqe                     # separates a file in varius datafames for data handling
    get_directory                       # Opens windows to let you search for a directory
    eqeFile                             # Checks id the file is an EQE and prepares the data: need to be modified for different files
    extractIDAndCell                    # Extracts the ID and cell number from a file: needs to be modified for different files
    plot_graph                          # Plots one or many lists to graphs in one figure
    plot_graphs                         # Plots two lines with different properties and adds vertical lines to given values

"""
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

#string_jsc = "short-circuit current density"  # for extraction of the Jsc values from file


def find_string_in_list(list, string):
    for index, line in enumerate(list):
      if string in line:
        return index
    return -1

def get_values_around_index(df, index, range):
    """Gets 5 values before and after the given index from a pandas dataframe."""
    range = int(range)
    before = df.iloc[index - range:index]  # Get the 5 values before the given index.
    after = df.iloc[index:index + range]  # Get the 5 values after the given index.
    values = pd.concat([before, after])  # Combine the before and after values into a single dataframe.
    #print("before: ", before,"        after: ",after)
    values_before = df.iloc[:(index - range)]
    values_after = df.iloc[(index + range):]
    return values, values_before,values_after

def interpolate_values(values):
    """Interpolates the values around the given index by adding num_steps."""
    value_x = values["wavelength in nm"]
    values_y = values["EQE at 0.000V"]
    #print("_____________value_x__________________")
    #print(value_x)
    #print("_______________values_y________________")
    #print(values_y)
    interpolator = interp1d(value_x, values_y)  # Create an interpolator for the values.
    start_value = values["wavelength in nm"].iloc[0]  # Get the start and end values
    end_value = values["wavelength in nm"].iloc[-1]
    new_x_values = np.arange(start_value, end_value + 1, 1)  # creates a np array (from a, to b, with a step 1)
    #print("start_value: ",start_value,"    end_value: ",end_value)
    #print(new_x_values)
    integer_array = np.array(new_x_values)
    new_y_values = interpolator(new_x_values)  # Interpolate the y values.
    interpolated_values = pd.DataFrame({"wavelength in nm": new_x_values, "EQE at 0.000V": new_y_values})  # Add the new values to the pandas dataframe.
    #print("______________interpolated_values_________________")
    #print(interpolated_values)
    return interpolated_values

def get_better_range(df, column_name):
    """        Finds the total count of numbers that are spaced by 1 within the given column of the dataframe."""
    count = 1
    count_of_consecutive_numbers = []
    for i in range(len(df)):
        if i == 0:
            continue
        if df[column_name][i] == df[column_name][i - 1] + 1:
            count += 1
        else:
            if count > 1:
                count_of_consecutive_numbers.append(count)
            count = 1
    if count > 1:
        count_of_consecutive_numbers.append(count)
    return sum(count_of_consecutive_numbers)

def get_slope(values):
    x_array = values["wavelength in nm"].values.reshape([-1, 1])
    y_array = values["EQE at 0.000V"].values.reshape([-1, 1])
    model = LinearRegression()
    model.fit(x_array, y_array)
    slope = model.coef_
    intercept = model.intercept_
    slope2 = model.coef_
    #print('slope: ', slope)
    intercept2 = model.intercept_
    #print('intercept: ', intercept)
    min_x_intercept = -intercept/slope
    #print('min x_intercept_1: ', min_x_intercept)
    slope = model.coef_.tolist()
    intercept = model.intercept_.item()
    return slope[0][0],intercept, min_x_intercept

def is_monotonic_increasing(df):
  step_differences = []
  for i in range(len(df) - 1):
    step_differences.append(df.iloc[i + 1] - df.iloc[i])
  return len(set(step_differences)) == 1

def get_minimum_derivative_index(df,min_eqe):
    """Gets the index of the minimum value of the first derivatives with respect to x in a pandas dataframe.  """
    global counter
    global dydx_a
    counter += 1
    # Calculate the first derivative of the y values with respect to x ##According to bard
    if is_monotonic_increasing(df["wavelength in nm"]):
        difStr = "Differential_1"+'_' + id_string + '_Cell ' + cell_number
        # The function is monotonically increasing, so use the '.diff()' method to calculate the derivative
        dif_calc = df["EQE at 0.000V"].diff()
        dydx = pd.DataFrame(
            {"wavelength in nm": df["wavelength in nm"], "EQE at 0.000V": df["EQE at 0.000V"], difStr: dif_calc})
        #dydx_a.insert((1), difStr, dif_calc)
        min_index = int(dydx[difStr][20:].idxmin())  # Find the index of the minimum value of the first derivative.
        dydx["Min_Eg" + difStr] = df["wavelength in nm"][min_index]
        min_eqe = dydx["Min_Eg" + difStr][2]
        if plot_derivative_method ==1:
            plot_graph(difStr, 'nm', 'Differential', dydx['wavelength in nm'], dydx[difStr])
    else:
        # The function is not monotonically increasing, so calculate the derivative using a custom method
        #print("Wavelengths are not equidistant")
        difStr = "Differential_2"+'_' + id_string + '_Cell ' + cell_number
        derivative = [0]
        for i in range(len(df) - 1):
            derivative.append((df['EQE at 0.000V'][i + 1] - df['EQE at 0.000V'][i]) / (
                        df['wavelength in nm'][i + 1] - df['wavelength in nm'][i]))

        derivative2 = pd.DataFrame(df, columns=['wavelength in nm','EQE at 0.000V'])
        derivative2.insert(2, difStr, derivative)  ##insert array 'derivative' in the dataframe derivative2, in column 2; with title difStr
        better_range = int((get_better_range(derivative2, "wavelength in nm"))/2)
        index_current = derivative2.loc[df["wavelength in nm"] == min_eqe].index[0]
        min_index = derivative2.loc[index_current - better_range:index_current + better_range, difStr].idxmin()
        derivative2["Min_Eg" + difStr] = df["wavelength in nm"][min_index]
        min_eqe = derivative2["Min_Eg" +difStr][2]
        if plot_derivative_method ==1:
            plot_graphs(difStr, 'nm', 'Differential', df['wavelength in nm'], derivative2)
    #print("The minimum content of wavelength in nm within the range of indexes  is:", min_eqe)
    return min_index, min_eqe


def file_separation_IV(newFile):  ##should open a file containing the list of movements
    with open(newFile, "r") as f:
        content = f.readlines()
    #string = "voltage"
    content = [line.replace('\n', '') for line in content]  # Remove the newline characters from the content.
    name_n_area = content[0]
    ID_n_cell = name_n_area.split(':')[1]
    area = content[0].split(':')[1]
    name_ID = ID_n_cell.split('cell')[0]
    cell =(ID_n_cell.split('cell')[1]).split('\t')[0]
    area = content[0].split(':')[2]
    name_ID_n_cell_no = area[1]

    #print('name_ID: ',name_ID,'; #cell#',cell, '; area: ', area)
    column_names = content[1]  # Get the column names from the first row of the list.
    column_names= column_names.split('\t')
    List1 = content[2:] # Remove the first rows from the list.
    List1 = [line.split('\t') for line in List1]  # splits content
    df_data = pd.DataFrame(List1, columns=column_names).astype(float)  # Create a DataFrame from the list data.
    df_data.insert(0, 'cell#',cell)  ##insert array 'sample_ID' in the dataframe df_data, in column 0; with title Sample
    df_data.insert(0, 'Sample',name_ID)  ##insert array 'sample_ID' in the dataframe df_data, in column 0; with title Sample
    df_data['area']= area

    #print(df_data.to_string())
    return df_data

def file_separation_IV_parameters(newFile):  ##should open a file containing the list of movements
    with open(newFile, "r") as f:
        content = f.readlines()
    content = [line.replace('\n', '') for line in content]  # Remove the newline characters from the content.
    #print(content)
    string = "max"
    sample_ID = newFile.split('Parameter')[0]
    #print('ID_of sample: ', sample_ID)
    max_min_index = find_string_in_list(content, string)  #  to get the index where the satistic values are found
    List1 = content[0:(max_min_index - 1)]
        #To get the IV parameters of each cell
    #print('name_n_area ',name_n_area)
    column_names = List1[0]  # Get the column names from the first row of the list.
    #print('column_names ',column_names)

    column_names= column_names.split('\t')
    List1 = List1[1:] # Remove the first rows from the list.
    List1 = [line.split('\t') for line in List1]  # splits content

    List2 = content[max_min_index:-1]  # to add statistics from that sample:
    List2 = [line.split('\t') for line in List2]  # splits content
#
    df_data = pd.DataFrame(List1, columns=column_names)#.astype(float)  # Create a DataFrame from the list data.
    df_data.insert(0, 'Sample',sample_ID)  ##insert array 'sample_ID' in the dataframe df_data, in column 0; with title Sample

    #Columns are the same for the statistics values but no temp
    column_names.remove("T start in °C")#delete this column
    column_names.remove("T end in °C")#delete this column
    #replace name of column
    index = column_names.index("cell")
    column_names[index] = "Stats"

    statistics_IV = pd.DataFrame(List2, columns=column_names).astype(str)  # Create a DataFrame from the list data.
    statistics_IV.insert(0, 'Sample',sample_ID)  ##insert array 'sample_ID' in the dataframe df_data, in column 0; with title Sample
    #print(statistics_IV.to_string())

    return df_data, statistics_IV

def file_separation_eqe(newFile):  ##should open a file containing the list of movements
    with open(newFile, "r") as f:
     content = f.readlines()
    content = [line.replace('\n', '') for line in content]  # Remove the newline characters from the content.
    comments_index = find_string_in_list(content, string_jsc)
    #print(content)
    column_names = content[0] # Get the column names from the first row of the list.
    #print(column_names)
    List1 = content[0:(comments_index-1)]
    #print(List1)
    List2 = content[comments_index:]
    #print(List2)
    column_names= column_names.split('\t')
    List1 = List1[1:] # Remove the first row from the list.
    List1 = [line.split('\t') for line in List1]  # splits content
    df_data = pd.DataFrame(List1, columns=column_names).astype(float)  # Create a DataFrame from the list data.
    #print(df_data)
    jsc_eqe = List2[0] # Get the column names from the first row of the list.
    #print(jsc_eqe)
    #column_names= column_names.split('\t')
    comments = List2[1:] # Remove the first row from the list.
    #List2 = [line.split('\t') for line in List2]  # splits content
    #comments = pd.DataFrame(List2, columns=column_names) # Create a DataFrame from the list data.
    return df_data, comments,jsc_eqe

def get_directory():
    directory = askdirectory()
    path = os.path.join(directory)
    print(path)
    #window = tk.Tk()
    #button = tk.Button(window, text="Browse Directory", command=browse_directory)
    #button.pack()
#
    #window.mainloop()
    return path

def IV_file(working_directory):  ##adjust the info in the file to be handeld in a dataframe
    hell_dat_files = []    # Create a list of .dat files in the current working directory.
    dunkel_dat_files = []    # Create a list of .dat files in the current working directory.
    parameter_dat_files = []    # Create a list of .dat files in the current working directory.
    dataframes_hell = {}    # Create a dictionary to store the DataFrames.
    dataframes_dunkel = {}    # Create a dictionary to store the DataFrames.
    dataframes_parameter = {}    # Create a dictionary to store the DataFrames.
    statistics_IV = {}

    for file in os.listdir(working_directory):
        if file.endswith(".dat") and 'hell' in file:
            hell_dat_files.append(file)
        elif file.endswith(".dat") and 'dunkel' in file:
            dunkel_dat_files.append(file)
        elif file.endswith(".dat") and 'arameter' in file:
            parameter_dat_files.append(file)
        else:
            print("Not possible with file: ", file)

        for dat_file in hell_dat_files:  # For each .dat file, create a DataFrame and store it in the dictionary.
            print("Light IV file: ", dat_file)
            df2 = file_separation_IV(dat_file)
            df2 = pd.DataFrame(df2)  # Create a DataFrame from the list of lists.
            df2["file_name"] = dat_file  # Save the file´s name in the DataFrame.
            df2['measurement_type'] = 'light'
            dataframes_hell[dat_file] = df2  # Add the DataFrame to the dictionary.
        ## fitting of IV to be included here

        for dat_file in dunkel_dat_files:  # For each .dat file, create a DataFrame and store it in the dictionary.
            print("Dark IV file: ", dat_file)
            df2 = file_separation_IV(dat_file)
            df2 = pd.DataFrame(df2)  # Create a DataFrame from the list of lists.
            df2["file_name"] = dat_file  # Save the file´s name in the DataFrame.
            df2['measurement_type'] = 'dark'
            dataframes_dunkel[dat_file] = df2  # Add the DataFrame to the dictionary.
        ## fitting of IV to be included here

        for dat_file in parameter_dat_files:  # For each .dat file, create a DataFrame and store it in the dictionary.
            print("Parameter file: ", dat_file)
            df3, df4 = file_separation_IV_parameters(dat_file)
            dataframes_parameter[dat_file] =df3
            statistics_IV[dat_file]  =df4
            #df2["file_name"], statistics_IV["file_name"] = dat_file  # Save the file´s name in the DataFrame.
            #df2['measurement_type'] = 'parameter'
            #dataframes_parameter[dat_file] = df2  # Add the DataFrame to the dictionary.
            #statistics_IV[dat_file] = statistics_IV
    else:
        print("Not possible with file: ", dat_file)

    return dataframes_hell, dataframes_dunkel, dataframes_parameter, statistics_IV


def eqeFile(working_directory):  ##adjust the info in the file to be handeld in a dataframe

    dat_files = []    # Create a list of .dat files in the current working directory.
    for file in os.listdir(working_directory):
        if file.endswith(".dat") and file.startswith("EQE"):
            dat_files.append(file)
    string = "short-circuit current density"

    dataframes = {}    # Create a dictionary to store the DataFrames.
    for dat_file in dat_files:    # For each .dat file, create a DataFrame and store it in the dictionary.
        df_data, comments, jsc_eqe = file_separation_eqe(dat_file)
        df = pd.DataFrame(df_data)            # Create a DataFrame from the list of lists.
        df["file_name"] = dat_file            # Save the file´s name in the DataFrame.
        df["Jsc_eqe"] = jsc_eqe  # Add the DataFrame to the dictionary.
        dataframes[dat_file] = df            # Add the DataFrame to the dictionary.
    #for dat_file, df in dataframes.items():  # Print the DataFrames.
        #print(f"File name: {dat_file}")
        #print(dataframes)
    return dataframes

def extractIDAndCell(string):
    #string = "EQE of ID 2-219-4-13 cell 14 CIGS.dat"
    # Extract the string between 'ID' and 'cell'
    id_string = re.search(r"ID\s+(.+?)\s+cell", string)
    if id_string is not None:
        id_string = id_string.group(1)
    else:
        id_string = ""

    # Extract the number between 'cell' and 'CIGS'
    cell_number = re.search(r"cell\s+(.+?)\s+CIGS", string)
    if cell_number is not None:
        cell_number = cell_number.group(1)
    else:
        cell_number = ""
    return id_string,cell_number


def smoothen_EQE_derive(df2,a=3):
    x = df2.loc[0]
    y = df2.loc[1]

    w = np.ones(a)/a        # Create a moving average filter with window size 'a'
    # Smooth the data
    smooth_y = np.convolve(y, w, mode='same')
    # Plot the data
    plot_comparison=0
    if plot_comparison ==1:
        plt.plot(x, y, label='Original data')
        plt.plot(x, smooth_y, label='Smoothed data',linestyle="dashed", alpha=0.3)
        plt.legend()
        #plt.savefig()
        plt.show()

        new_curve = pd.DataFrame({"x": x, "y": smooth_y})
        return new_curve

def include_variations(df, compare_variations, compare_IDs):
    if len(compare_variations) == len(compare_IDs):
        df['Variations'] = df['Sample'].apply(lambda x:compare_variations[compare_IDs.index(x)] if x in compare_IDs else 'NA')
    else:
        print('The arrays variations and samples are not the same. Please check again!')
        print(compare_IDs.to_string())
        print(compare_IDs.to_string())
        print(df.to_string())
    return df
