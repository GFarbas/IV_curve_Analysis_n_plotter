import numpy
import numpy as np
import pandas as pd
import pvlib
import pvlib.ivtools
import datetime
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import Plotting_functions

date_object = datetime.date.today()
from pvlib import pvsystem, ivtools

temperature = 298  # temp in K
boltzmannOverCharge = 8.61733E-05  # Boltzmann constant over q in eV/K
noOfCells = 1
q = 1.67E-19

# plot_ivs = 0
def sandia_fit(voltage, current, noOfCells=1):
    ''' SANDIA fit uses the light IV to extract the parameters
    Only works for the firs quadrant of the IV
    Thus only positive values f voltage and current'''
    print('-----------------sandia_fit-----            -----------')
    voltage, current = make_iv_fwd_n_positive(voltage, current)
    parameters = IV_parameter_extraction(voltage, current)
    #parameters from IV : voc, jsc, pmpp,ff,vmpp,impp
    ##make an array from Voc to Jsc for voltage and current
    sandia_fit_results=[]
    voc = parameters[0]
    jsc= parameters[1]
    v_mp_i_mp_tup = (parameters[4],parameters[5])
    voltage, current = include_voc_in_iv(voltage, current, voc)

    voltage_sde, current_sde = iv_only_quadrant_1(voltage, current)  ##saves array from jsc to voc as numpy
    tup_titles = ('I_ph           ', 'I_0          ', 'Rsh              ', 'Rs          ', 'nNsVth')
    #print(dir(pvlib.ivtools))  #prints all atributes
    tuple_results_a = pvlib.ivtools.sde.fit_sandia_simple(voltage_sde, current_sde,v_oc=voc, i_sc=jsc, v_mp_i_mp=v_mp_i_mp_tup,
                      vlim=0.2, ilim=0.1)  # Makes fit saves it as txt
    SDE_Sandia_fit = numpy.array(tuple_results_a, dtype=numpy.float32)  # Transforms array list into float
    photocurrent = SDE_Sandia_fit[0]
    saturation_current = SDE_Sandia_fit[1]
    resistance_series = SDE_Sandia_fit[2]
    resistance_shunt = SDE_Sandia_fit[3]
    nNsVth = SDE_Sandia_fit[4]
    ThermV = temperature * boltzmannOverCharge
    idealityFactor = nNsVth / (noOfCells * ThermV)
    # Output-----
    print_results = 1
    if print_results == 1:
        print(("-----------SDE Sandia Fit---------------------"))
        print((tup_titles))
        print((SDE_Sandia_fit))
        print('Jsc= ' + str(photocurrent))
        print('Io= ' + str(saturation_current))
        print('n= ' + str(idealityFactor))
        print('Rs= ' + str(resistance_series) + ' [ohm]')
        print('Rp= ' + str(resistance_shunt) + ' [ohm]')
        print(("--------------------------------"))
        print(("--------------------------------"))
        print(("--------------------------------"))

    sandia_fit_results = [photocurrent, saturation_current, idealityFactor, resistance_series, resistance_shunt]
    print('-----------------sandia_fit--out--------------')

    return sandia_fit_results

def dark_IV_linear_extraction(voltage, current, Voc=600, noOfCells=1):
    '''Takes Voc as starting point to make a linear regression '''
    print('Dark IV lienar extraction')
    photocurrent = 0
    saturation_current = 0
    resistance_series = 0
    resistance_shunt = 0
    nNsVth = 0
    ThermV = 0
    idealityFactor = nNsVth / (noOfCells * ThermV)
    dark_IV_diode_parameters= [photocurrent, saturation_current, idealityFactor, resistance_series, resistance_shunt]
    print('Dark IV extraction')
    return dark_IV_diode_parameters

def dark_IV_fit(voltage, current, Voc=600, noOfCells=1):
    '''Fit the diode equation to the data '''
    params, _ = optimize.curve_fit(
        diode_equation_dark, voltage, current, p0=[1e-12, 1.5, 10000, 20, 300, 8.61733E-05 ])

    # Print the saturation current and the ideality factor
    print("dark_IV_fit")
    print("Saturation current:", params[0])
    print("Ideality factor:", params[1])
    print("Series resistance:", params[5])

    # Fit the diode equation to the data
    '''Takes Voc as starting point to make a linear regression '''
    print('Dark IV extraction')
    photocurrent = 0
    saturation_current = params[0]
    resistance_series = params[5]
    resistance_shunt =params[3]
    #nNsVth = 0
    #ThermV = 0
    idealityFactor = params[1] #nNsVth / (noOfCells * ThermV)
    dark_IV_diode_parameters= [photocurrent, saturation_current, idealityFactor, resistance_series, resistance_shunt]
    voltage_fit = np.linspace(0, 1, 1000)
    current_fit = diode_equation_dark(voltage_fit, *params)
    #dark_iv_fit_data = pd.DataFrame({'voltage_dark_fit':voltage_fit, 'current_dark_fit':current_fit})
    Plotting_functions.plot_iv('dark IV fit','voltage','current',voltage, current)
    Plotting_functions.plot_iv('dark IV fit','voltage','current',voltage_fit, current_fit)
    print("dark_IV_fit")

    return dark_IV_diode_parameters


def diode_equation_dark(voltage, Is, n, Rsh, Rs, T=300,  k=8.61733E-05):
    k= boltzmannOverCharge
    current = np.zeros_like(voltage)
    current_a =Is * np.exp(((q * (voltage - current * Rs)) /(n*k*T))-1) - (voltage - current * Rs) / Rsh
    return current_a

def diode_equation_light(voltage, Is, n, Rsh, Rs,current_photo=0, T=300,  k=8.61733E-05):
    k= boltzmannOverCharge
    current = np.zeros_like(voltage)
    current_a = current_photo - Is * np.exp(((q * (voltage - current * Rs)) /(n*k*T))-1) - (voltage - current * Rs) / Rsh
    return current_a


def IV_parameter_extraction(voltage, current):
    print('IV_parameters_interpolated_ self made')
    voltage, current = make_iv_fwd_n_positive(voltage, current)
    IV_parameters_interpolated = np.zeros(6)
    #Check if Voc already measured
    jsc = float(current[voltage == 0])
    print('The Jsc measured at V=0:', jsc)
    voc = voltage[current == 0]
    if voc.empty:
        print('No Voc in file')
        ##get voltage around Jsc +- 5mA
        voltage_for_voc_extraction = voltage[current.between(-8, 8)]
        current_for_voc_extraction = current[current.between(-8, 8)]
        #extract Voc
        interpolator_voc = interp1d(current_for_voc_extraction,voltage_for_voc_extraction)  # Create an interpolator for the values.
        voc = float(interpolator_voc(0))  # Interpolate the y values.
    pmpp = float(max(voltage*current))
    ff = float(pmpp/(voc*jsc))
    impp = float(current[(voltage*current)== pmpp])
    vmpp = float(voltage[(voltage*current)== pmpp])

    IV_parameters_interpolated[0] = voc
    IV_parameters_interpolated[1] = jsc
    IV_parameters_interpolated[2] = (pmpp)
    IV_parameters_interpolated[3] = (ff)
    IV_parameters_interpolated[4] = (vmpp)
    IV_parameters_interpolated[5] = (impp)
    #parameters from IV : voc, jsc, pmpp,ff,vmpp,impp
    print('parameters from IV :')
    print('voc: ',round(voc,4), 'jsc: ', jsc,'pmpp: ', round(pmpp,3),'ff: ', round(ff,3),'vmpp: ',\
          round(vmpp,3),'impp: ', (impp,3))

    return IV_parameters_interpolated  #


def make_iv_fwd_n_positive(voltage, current):
    # flip the dataframe if saved in reverse bias
    if voltage[0] > voltage[10]:  #Check if the df is going from small to large numbers
        voltage_fwd = voltage.iloc[::-1]
        current_fwd = current.iloc[::-1]
        #print(voltage_fwd)
        voltage = voltage_fwd.reset_index(drop=True) #resets index values
        current = current_fwd.reset_index(drop=True)  # resets index values
        j_at_V = float(current[voltage == 0.2])  ##get J values around Voc= 0.2
        #print(j_at_V)
        if j_at_V < 0:
            current = current*-1  # resets index values
        #print(voltage_fwd)
    #plt.plot(voltage, current)
    #plt.show()
   #print(voltage.to_string())
    #print(current.to_string())
    return voltage, current


def iv_only_quadrant_1(voltage,current):
    print('----------iv_only_quadrant_1--------')
    iv =pd.DataFrame()
    iv['voltage'] = voltage
    iv['current'] = current
    iv =iv[iv>=0].dropna() #drop Nan values
    voltage = numpy.array(iv['voltage'], dtype=numpy.float32)  #Save it a numpy array
    current =numpy.array(iv['current'], dtype=numpy.float32)
    print('----------iv_only_quadrant_out--------')
    return voltage, current


def include_voc_in_iv(voltage,current,voc):
    print('include_voc_in_iv')
    voltage = pd.DataFrame({'voltage': voltage })
    #current = pd.DataFrame({'current': current})
    #current = list(current)
    iv =pd.DataFrame()
    #print(voltage.to_string())
    #print(current.to_string())
    voltage.loc[len(voltage)] = voc  #include Voc
    current.loc[len(current)] = 0.0  # include J value for Voc
    # Sort the voltage dataframe and reset indexes
    iv['voltage'] = voltage #add to df iv
    iv['current'] = current #add to df iv
    iv = iv.sort_values(by=['voltage'], ascending=True).reset_index(drop=True) ## sort df iv using the voltage column
    #print(iv.to_string())
    voltage=iv['voltage']  #Save it as Series
    current=iv['current']  #Save it as Series
    print('include_voc_in_out')

    return voltage, current