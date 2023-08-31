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
    voltage, current = make_iv_fwd_n_positive(voltage, current)
    parameters = IV_parameter_extraction(voltage, current)
    print(parameters)
    ##make an array from Voc to Jsc for voltage and current
    voc =1
    voltage_sde = numpy.array(voltage[voltage[0 <= voltage <= voc]], dtype=numpy.float32)
    current_sde = numpy.array(current[current[voltage[0 <= voltage <= voc]]], dtype=numpy.float32)
    tup_titles = ('I_ph           ', 'I_0          ', 'Rsh              ', 'Rs          ', 'nNsVth')
    #print(dir(pvlib.ivtools))  #prints all atributes

    tuple_results_a = pvlib.ivtools.sde._sandia_simple_params(voltage_sde, current_sde, v_oc=None, i_sc=None, v_mp_i_mp=None,
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
    print_results = 0
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
    return sandia_fit_results

def dark_IV_extraction(voltage, current, Voc=600, noOfCells=1):
    '''Takes Voc as starting point to make a linear regression '''
    print('Dark IV extraction')
    photocurrent = 0
    saturation_current = 0
    resistance_series = 0
    resistance_shunt = 0
    nNsVth = 0
    ThermV = 0
    idealityFactor = nNsVth / (noOfCells * ThermV)
    dark_IV_diode_parameters= [photocurrent, saturation_current, idealityFactor, resistance_series, resistance_shunt]
    return dark_IV_diode_parameters

def dark_IV_fit(voltage, current, Voc=600, noOfCells=1):

    # Fit the diode equation to the data
    params, _ = optimize.curve_fit(diode_equation, voltage, current, p0=[Is, n, k, T, Rsh])
    '''Takes Voc as starting point to make a linear regression '''
    print('Dark IV extraction')
    photocurrent = 0
    saturation_current = 0
    resistance_series = 0
    resistance_shunt = 0
    nNsVth = 0
    ThermV = 0
    idealityFactor = nNsVth / (noOfCells * ThermV)
    dark_IV_diode_parameters= [photocurrent, saturation_current, idealityFactor, resistance_series, resistance_shunt]
    return dark_IV_diode_parameters

def diode_equation_dark(voltage, Is, n, Rsh, Rs, T=300,  k=8.61733E-05):
    k= boltzmannOverCharge
    current = current = np.zeros_like(voltage)
    current_a =Is * np.exp(((q * (voltage - current * Rs)) /(n*k*T))-1) - (voltage - current * Rs) / Rsh
    return current_a

def plot_dark_fit(voltage,current):
    # Fit the diode equation to the data
    params, _ = optimize.curve_fit(
        diode_equation_dark, voltage, current, p0=[1e-12, 1.5, 10000, 20, 300, 8.61733E-05 ])

    # Print the saturation current and the ideality factor
    print("Saturation current:", params[0])
    print("Ideality factor:", params[1])
    print("Series resistance:", params[5])

    # Plot the data and the fitted curve
    plt.plot(voltage, current, "o")
    voltage_fit = np.linspace(0, 1, 1000)
    current_fit = diode_equation_dark(voltage_fit, *params)
    plt.plot(voltage_fit, current_fit)

    # Label the axes
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")

    # Show the plot
    plt.show()
    return params, _

def IV_parameter_extraction(voltage, current):
    print('IV_parameters_interpolated not ready')
    voltage, current = make_iv_fwd_n_positive(voltage, current)
    IV_parameters_interpolated = np.zeros(6)

    #Check if Voc already measured
    jsc = current[voltage == 0]
    print('The Jsc measured at V=0:', jsc)

    voc = voltage[current == 0]
    print(voc)
    if voc.empty:
        print('No Voc in file')
        ##get voltage around Jsc +- 5mA
        voltage_for_voc_extraction = voltage[current.between(-8, 8)]
        print('voltage_for_voc_extraction')
        print(voltage_for_voc_extraction)
        current_for_voc_extraction = current[current.between(-8, 8)]
        print('current_voc')
        print(current_for_voc_extraction)
        #extract Voc
        interpolator_voc = interp1d(current_for_voc_extraction,voltage_for_voc_extraction)  # Create an interpolator for the values.
        voc = float(interpolator_voc(0))  # Interpolate the y values.
        print('voc: ', voc)
        print('jsc: ', jsc)
    pmpp = max(voltage*current)
    ff = pmpp/(voc*jsc)
    impp = current[(voltage*current)== pmpp]
    vmpp = voltage[(voltage*current)== pmpp]

    IV_parameters_interpolated[0] = voc
    IV_parameters_interpolated[1] = jsc
    IV_parameters_interpolated[2] = pmpp
    IV_parameters_interpolated[3] = ff
    IV_parameters_interpolated[4] = vmpp
    IV_parameters_interpolated[5] = impp

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
    print(voltage.to_string())
    print(current.to_string())
    return voltage, current

