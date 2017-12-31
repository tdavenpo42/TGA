# -*- coding: utf-8 -*-
# equil_nonstoich_from_TGA.py

# Required modules: sys.exit for breaking script when there is a data read
# failure; several os functions to handle folder and file management;
# copyfile for file copying; rmtree for removing folder and contents;
# numpy for handling calculations; floor for rounding to integer indices; 
# pyplot for plotting; Image from PIL for handling TIFF compression; 
# stats for calculating confidence interval; curve_fit for fitting mass 
# decay to a function
from sys import exit as sysexit
from os import mkdir, remove, path
from os.path import exists, expanduser
from shutil import copyfile, rmtree
from platform import system
from re import split
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from scipy.optimize import curve_fit

# Set plotting style - will need to add publication.mplstyle to style library 
plt.style.use('publication')

# Expected errors of standard measurements
sample_mass_err = 0.5 # Error of measuring sample mass in mg
bg_sub_err = 0.3 # Error of background subtraction in mg

# Define fitting function for nonequilibrium mass curve
def exp_fit(T,a,tau,m_eq):
    t,t0 = T
    return a * np.exp(-tau * (t-t0)) + m_eq

# Molar masses of relevant species of species
M_O = 15.9994; M_MOx = 0. #M_MOx determined from file

# Intialize flags for sample mass presence and sample mass unit type
sample_mass_mg_flag, mass_given_flag = False, False
# Initialize flags for mass list unit type
mass_pct_flag, mass_mg_flag = False, False
# Initialize flags for known composition
known_composition_flag = False

#Initialize lists for TGA data
temp, time, mass_loss, segment = [],[],[],[]

## Read in TGA data file, determine if sample composition and mass are known,
## return lists of molecular mass, temp, time, mass_loss, and temp segment 

#filename = input('Input file name without .txt extension: ')
filename = '130606_10Pr_10pctO2'
ptm = open(filename + '.txt')

# Create the destination directory for data output
userhome = expanduser('~')

# Path to Desktop on Windows
if system() == 'Windows':
    userhome = userhome.replace('\\', '/') + '/Desktop/'

# Path to Desktop on Mac
if system() == 'Darwin':
    userhome = userhome + '/Desktop/'
    
savedir = userhome+filename+'_analyzed'
if not exists(savedir):
    mkdir(savedir)

# Make directory for fitting results - delete previous version if exists.  
# Otherwise program will add additional fit figures to previous ones instead of
# overwriting.
fitdir = savedir + '/' + 'fits'
if exists(fitdir):
    rmtree(fitdir)
    mkdir(fitdir)
else:
    mkdir(fitdir)

# Output a copy of the source file in the destination folder
copyfile(filename+'.txt',savedir+'/'+filename+'.txt')

# Read in lines of data file
for line in ptm: 

    # Check if line is a header line
    if line.startswith('#'):
 
        # For header lines: extract sample composition
        if line.startswith('SAMPLE:',1):
        
            #Check for handled sample compositions and set flags
            if line.split(':')[1].strip().startswith('10Pr'):
                M_MOx = 140.90765*0.1 + 140.116*0.9 + 15.9994*2
                known_composition_flag = True
            
            if line.split(':')[1].strip().startswith('LSM82'):
                M_MOx = 138.91*0.8 + 87.62*0.2 + 54.938*1 + 15.9994*3
                known_composition_flag = True
                
            if line.split(':')[1].strip().startswith('LuFeO3'):
                M_MOx = 174.967*1 + 55.845*1 + 15.9994*3
                known_composition_flag = True
            
        # For header lines: extract sample mass
        elif line.strip().startswith('SAMPLE MASS',1):
            if 'mg' in line:
                mass = float(line.split(':')[1])
                sample_mass_mg_flag = True
                if mass != 0.: mass_given_flag = True
    
    # Check for list label line
    elif line.startswith('##'):
                           
        # Check if mass is in % or in mg
        if '%' in line: mass_pct_flag = True
        elif 'mg' in line: mass_mg_flag = True
        else: print('Unknown mass units')
                    
    # Other lines are data list lines
    else:
        # Ignore empty lines
        line = line.strip()
        if line:
            # split line into fields - semicolon or whitespace delimited
            fields = split('[;\s]+',line)
                  
            # add fields to data lists
            temp += [float(fields[0])]
            time += [float(fields[1])]
            mass_loss += [float(fields[2])]
            segment += [int(fields[-1])]
              
# Request molar mass of composition if not determined from file       
if not known_composition_flag:
    M_MOx = float(input('Sample composition unknown. Input molar \
mass of composition in g/mol: '))
    if M_MOx != 0: known_composition_flag = True

# Request sample mass if not read from file        
if not mass_given_flag:
    mass = float(input('Sample mass unknown. Input sample \
mass in mg: '))
    if mass != 0: mass_given_flag = True

# Convert data lists to arrays
temp = np.array(temp)
time = np.array(time)
mass_loss = np.array(mass_loss)
segment = np.array(segment)

# If mass_loss given in %, convert to mg
if mass_pct_flag: 
    mass_loss = mass*mass_loss/100 - mass

# Check that we have extracted sample mass and oxide molecular mass 
if not mass_given_flag or not known_composition_flag:
    sysexit('Sample mass or molecular weight unreadable from ' + filename)
    
## Organize TGA data by temperature program segments and identify 
## isothermal segments

# Determine a matrix of rows of segment, start index of that segment, end 
# index of that segment, data points per minute of segment, ramp rate of
# segment, isothermal segment flag

# Determine number of segments in file
seg_num = np.nanmax(segment)
# Preallocate segment index matrix
seg_index = np.zeros((seg_num,6))
# Determine index values for each segment
for i in range(seg_num):
    # Segment number
    seg_index[i,0] = i+1
    # Determine start and end index of segment number
    seg_args = np.argwhere(segment==i+1)
    seg_index[i,1] = seg_args[0]
    seg_index[i,2] = seg_args[-1]
    # Determine # of data points per minute: can only be input as integer
    seg_index[i,3] = np.around((seg_args[-1]-seg_args[0])/(time[seg_args[-1]]
    -time[seg_args[0]]))
    
    # Determine if segment is isothermal: take the last five minutes of data
    # (early points in isothermal segments can be misleading)
    # and determine the average ramp rate over that time

    # If there are more than five minutes of data
    if seg_args[-1] - seg_args[0] >= seg_index[i,3]*5:
    # Determine ramp_rate
        seg_index[i,4] = (temp[seg_args[-1]]-
                          temp[seg_args[-1-int(seg_index[i,3])*5]])/5
        # If segment ramp rate is less than 0.1 C/min assume isothermal
        if np.absolute(seg_index[i,4]) < 0.1:
            # Set isothermal flag
            seg_index[i,5] = 1
    # Assume that a segment shorter than 5 minutes is not isothermal
    else:
        # Determine ramp rate
        seg_index[i,4] = (temp[seg_args[-1]]-temp[seg_args[0]]
                          )/(time[seg_args[-1]]-time[seg_args[0]])

## Determine equilibrium temperature and mass for isothermal segments

# Preallocate matrix of rows of equilibrium temperature and equilibrium mass
# by average, with mass equilibration flag, 95%CI of mass,
# amount of time over which equilibrium is calculated, start and end times,
# predicted equilibrium mass from curve fitting, 95%CI of fit mass, fit 
# reliability flag
eq_vals = np.zeros((np.count_nonzero(seg_index,axis=0)[5],11))

# For isothermal segments in seg_index
j = 0
for i in range(seg_index.shape[0]):
    if seg_index[i,5] == 1:
        # Determine equilibrium temperature using last half of segment data
        # Determine number of points in half the segment
        half_seg = floor((seg_index[i,2]-seg_index[i,1]+1)/2)
        eq_vals[j,0] = np.average(temp[half_seg+int(seg_index[i,1])
                                       :int(seg_index[i,2])])
        ## Determine best equilibrium mass
        ## Logic: mass should be within tolerance of mass change rate for at
        ## least 5 minutes to be considered at equilibrium. Ideally, want to
        ## take as much data as possible for the mean up to half of the 
        ## segment data.
        
        # Test for equilibrium mass on half segment
        test_mass_slice = mass_loss[half_seg+int(seg_index[i,1])
                                    :int(seg_index[i,2])]
        # Test mass change rate on average of first and last minute of 
        # considered data
        smooth_mass_beg = np.average(test_mass_slice[0:int(seg_index[i,3])])
        smooth_mass_end = np.average(test_mass_slice[-1-int(seg_index[i,3])
                                     :-1])
        rate_mass_change = np.abs((smooth_mass_end-smooth_mass_beg)*int(
                seg_index[i,3])/(len(test_mass_slice)-int(seg_index[i,3])))
        # If mass change rate is < 0.001 mg/min consider at equilibrium
        if rate_mass_change < 0.001:
            # Add equilibrium mass
            eq_vals[j,1] = np.average(test_mass_slice)
            # set equilibrium flag
            eq_vals[j,2] = 1
            # Calculate 95% confidence interval
            eq_vals[j,3] = stats.sem(test_mass_slice)*stats.t.ppf(
                                     1-0.05,len(test_mass_slice)-1)
            # Start time of averaged data
            eq_vals[j,5] = time[half_seg+int(seg_index[i,1])]
            # End time of averaged data
            eq_vals[j,6] = time[int(seg_index[i,2])]
            # Length of time considered for averaged mass
            eq_vals[j,4] = eq_vals[j,6]-eq_vals[j,5]

        # If standard deviation is too large on half of the data, check 
        # standard set of shorter time segments down to five minutes
        else:
            # Set standard time segments
            # Initialize time segment list
            time_seg = []
            # Counter
            k = 0
            # Generates list of number of data points in 5, 10, 20, 40 etc.
            # min, up to half of the data points in the segment 
            while (2**k)*5*seg_index[i,3] < half_seg:
                time_seg.insert(0,(2**k)*5*seg_index[i,3])
                k += 1
            # Use longest segment which results in a rate mass change < 0.001
            # mg/min
            for l in enumerate(time_seg):
                # Take last time_seg[l] of data
                test_mass_slice = mass_loss[int(seg_index[i,2])-int(time_seg[
                                            l[0]]):int(seg_index[i,2])]
                # Test mass change rate on average of first and last minute of 
                # considered data
                smooth_mass_beg = np.average(test_mass_slice[0:
                                             int(seg_index[i,3])])
                smooth_mass_end = np.average(test_mass_slice[-1-
                                             int(seg_index[i,3]):-1])
                rate_mass_change = np.abs((smooth_mass_end-smooth_mass_beg)*
                                         int(seg_index[i,3])/(len(
                                         test_mass_slice)-int(seg_index[i,3])))
                # Save data if rate mass change < 0.001 m/min or of last test
                # if mass change rate is too large for all tests
                if rate_mass_change < 0.001 or l[0] == len(time_seg)-1:
                    # Add equilibrium mass m_eq
                    eq_vals[j,1] = np.average(test_mass_slice)
                    # Set equilibrium flag only if there is a tested segment
                    # length that passes the mass change rate test
                    if rate_mass_change < 0.001:
                        eq_vals[j,2] = 1
                    # Calculate 95% confidence interval
                    eq_vals[j,3] = stats.sem(test_mass_slice)*stats.t.ppf(
                                             1-0.05,len(test_mass_slice)-1)
                    # Start time of averaged data
                    eq_vals[j,5] = time[int(seg_index[i,2])
                                        -int(time_seg[l[0]])]
                    # End time of averaged data
                    eq_vals[j,6] = time[int(seg_index[i,2])]
                    # Length of time considered for averaged mass
                    eq_vals[j,4] = eq_vals[j,6]-eq_vals[j,5]
                    # if mass change rate test is passed, dont consider 
                    # shorter time segments
                    break
                
        ## Determine best fit with exponential decay function
        # take half of segment to fit data        
        mass_slice = mass_loss[half_seg+int(seg_index[i,1]):
                               int(seg_index[i,2])]
        time_slice = time[half_seg+int(seg_index[i,1]):
                          int(seg_index[i,2])]
        # Determine start time of segment to be passed to fit function
        init_time = time[half_seg+int(seg_index[i,1])]
        # Set values to be passed to fit function
        Ts = [time_slice, init_time]
        # Define initial test values for fit function
        start_vals = [1,0.001,-1]
        # Fit curve with defined bounds: returns optimal values and covariance
        # matrix
        popt, pcov = curve_fit(exp_fit, Ts, mass_slice, p0=start_vals,
                     bounds =((-100,-1,-np.abs(100*mass_loss[
                              int(seg_index[i,2])])),(100,1,
                              np.abs(100*mass_loss[int(seg_index[i,2])]))),
                     max_nfev=1000)
        # Pass optimized m_eq to data matrix
        eq_vals[j,7] = popt[2]
        # Pass 95% confidence on m_eq to data matrix
        eq_vals[j,8] = np.sqrt(pcov[2,2])*stats.t.ppf(1-0.05,len(mass_slice)-1)
        
        ## Determine reliability of fit parameter: if fit value is too far
        ## from measured data, consider fit parameter unreliable
        # Difference between fit value and best equilibrium value
        eq_vals[j,9] = np.abs(eq_vals[j,7] - eq_vals[j,1])
        # Consider reliable if difference is < 0.5 mg
        if np.abs(eq_vals[j,9]) < 0.5:
            eq_vals[j,10] = 1 
        
        ## Make figure comparing slice data to best calculated average
        ## equilibrium mass and curve fit 
        if not (1450 < eq_vals[j-1,0] < 1550 and eq_vals[j,0] < 850):
            # Open a new figure object
            plt.figure(j) 
            # Label axes
            plt.xlabel('time (min)')
            plt.ylabel('mass change (mg)')
            # Plot raw data
            plt.plot(time_slice,mass_slice,'k')
            # Plot curve fit to data
            plt.plot(time_slice,exp_fit(Ts,*popt),'r')
            # Plot average equilibrium mass as horizontal line
            plt.axhline(eq_vals[j,1],color='b')
            # Round equilibrium temperature and save as string
            temp_str = str(int(round(eq_vals[j,0],0)))
            # Print equilibrium temperature and whether program determines 
            # that mass has reached equilibrium (thus using recorded average
            # mass value)
            # Average mass value used
            if eq_vals[j,2] == 1:
                plt.gcf().text(plt.gca().get_position().x1,
                    plt.gca().get_position().y1+0.015,'Temperature = '+
                    temp_str+' $^\circ$C\nequilibrium? Y', va='bottom',
                    ha='right')
            # Curve fit mass value used    
            else:
                plt.gcf().text(plt.gca().get_position().x1,
                    plt.gca().get_position().y1+0.015,'Temperature = '+
                    temp_str+' $^\circ$C\nequilibrium? N', va='bottom',
                    ha='right')
            # Save figure as tiff format to temp file
            
            # If there are multiple points at same temp, save as a second
            if path.isfile(fitdir+'/'+filename+'_'+temp_str+'.tiff'):
                m = 2
                while path.isfile(fitdir+'/'+filename+'_'
                               +temp_str+'_'+str(m)+'.tiff'):
                    m += 1
                plt.savefig(fitdir+'/'+filename+'_'+temp_str+'temp_'+str(m)
                            +'.tiff',dpi=300,bbox_inches='tight')
                # Convert uncompressed tiff to LZW compressed image
                im=Image.open(fitdir+'/'+filename+'_'+temp_str+'temp_'+str(m)+
                              '.tiff')
                im.save(fitdir+'/'+filename+'_'+temp_str+'_'+str(m)+'.tiff',
                        compression='tiff_lzw')
                # Close figure objects
                plt.close()
                im.close()
                # Delete temp file
                remove(fitdir+'/'+filename+'_'+temp_str+'temp_'+str(m)+'.tiff')
                     
            else:
                plt.savefig(fitdir+'/'+filename+'_'+temp_str+'temp.tiff',
                            dpi=300,bbox_inches='tight')
                # Convert uncompressed tiff to LZW compressed image
                im=Image.open(fitdir+'/'+filename+'_'+temp_str+'temp.tiff')
                im.save(fitdir+'/'+filename+'_'+temp_str+'.tiff',
                        compression='tiff_lzw')
                # Close figure objects
                plt.close()
                im.close()
                # Delete temp file
                remove(fitdir+'/'+filename+'_'+temp_str+'temp.tiff')
        # Increment equilibrium temperature step index
        j += 1

## Calculate and make array of equilibrium temperature and equilibrium
## nonstoichiometry using best guess equilibrium mass values, with 
## 95%CI of nonstoichiometry and give prediction for data reliability

#Initialize special exception flags
int_ref_flag = False

# Handle special exception: ignore isothermal at low temperature after 
# sequence of high temperatures - was used as a low temp anneal to return
# sample (mostly) to original state
if 1450 < eq_vals[-2,0] < 1550 and eq_vals[-1,0] < 850:
    # Preallocate array
    nonstoich_temp = np.zeros((eq_vals.shape[0]-1,3))
    
    # Acknowledge special case found
    print('\n\nLow temperature anneal step ignored.')

# Handle special exception: recognize experiment with internal reference state
elif np.abs(eq_vals[0,0] - eq_vals[-1,0]) < 5:
    
    # Acknowledge special case found
    print('\n\nRecognized internal reference at %.0f \u00b0C.' %(eq_vals[0,0]))
    
    # Set special case flag
    int_ref_flag = True
    
    # Determine mass shift to internal reference
    if eq_vals[0,2] == 1:
        mass_shift_init = eq_vals[0,1]
    else:
        mass_shift_init = eq_vals[0,7]
    if eq_vals[-1,2] == 1:
        mass_shift_final = eq_vals[-1,1]
    else:
        mass_shift_final = eq_vals[-1,7]
    mass_shift = (mass_shift_init + mass_shift_final)/2
    mass = mass + mass_shift
    
    # Shift masses
    eq_vals[:,1] -= mass_shift
    mass_loss -= mass_shift

    nonstoich_temp = np.zeros((eq_vals.shape[0],3))
# Regular data handling
else:
    nonstoich_temp = np.zeros((eq_vals.shape[0],3))

# Initialize list indicating results reliability
reliable = []

# Global calculations feeding into nonstoichiometry calculation  
# sample mass in moles
sample_mol = mass/1000/M_MOx
# sample mass error in moles
sample_mol_err = sample_mass_err/1000/M_O
# For each isothermal step

for i in range(nonstoich_temp.shape[0]):
    # take the equilibrium temperature
    nonstoich_temp[i,0] = eq_vals[i,0]
    # If equilibrium mass established
    if eq_vals[i,2] == 1:
        # oxygen loss in moles
        O_loss = -(eq_vals[i,1]/1000/M_O)
        # Calculate nonstoichiometry
        nonstoich_temp[i,1] = O_loss/sample_mol
        # Calcualte best estimate on error in nonstoichiometry
        # Use averaging error, error in sample mass, background subtraction 
        # error
        # oxygen loss masurement error in moles
        O_loss_err = np.sqrt(eq_vals[i,3]**2+bg_sub_err**2)/1000/M_O
        # Nonstoichiometry error
        nonstoich_temp[i,2] = np.abs(nonstoich_temp[i,1])*np.sqrt(
                (O_loss_err/O_loss)**2+(sample_mol_err/sample_mol)**2)
        # Denote value is reliable: equilibrium established
        reliable += 'Y'

    # If mass curve fit to exponential equation
    else:
        # oxygen loss in moles
        O_loss = -(eq_vals[i,7]/1000/M_O)
        # Calculate nonstoichiometry
        nonstoich_temp[i,1] = O_loss/sample_mol
        # Calcualte best estimate on error in nonstoichiometry
        # Use fit error, error in sample mass, background subtraction 
        # error
        # oxygen loss masurement error in moles
        O_loss_err = np.sqrt(eq_vals[i,8]**2+bg_sub_err**2)/1000/M_O
        # Nonstoichiometry error
        nonstoich_temp[i,2] = np.abs(nonstoich_temp[i,1])*np.sqrt(
                (O_loss_err/O_loss)**2+(sample_mol_err/sample_mol)**2)
        # Denote if value is reliable: if exp fit likely to be reliable
        if eq_vals[i,10] == 1:
            reliable += 'Y'
        else:
            reliable += 'N'

## Write equilibrium values to csv file
# Create matrix of strings to be output to csv file
output = np.column_stack((nonstoich_temp,reliable))
# Create csv file
np.savetxt(savedir+'/'+filename+'_output.csv',output,
           delimiter=',',header='Temperature (\u00b0C),nonstoichiometry,\
           nonstoichiometry error,reliable?', fmt='%s')

## Create figure visualizing raw data
# Create figure object
plt.figure()
# Label axes
plt.xlabel('time (min)')
if int_ref_flag == True:
    plt.ylabel('mass change ($m - m_0$, mg)')
else:
    plt.ylabel('mass change (mg)')
# Plot raw mass curve
plt.plot(time,mass_loss,'k')
# Create overlapping plot for temperature data
plt.twinx()
# Plot temperature curve with new y axes
plt.plot(time,temp,'b')
# Label new axis and color to distinguish from initial y axis
plt.ylabel('Temperature ($^\circ$C)',color='b')
plt.gca().spines['right'].set_color('b')
plt.gca().tick_params(axis='y', which='both',colors='b')
# Save figure as tiff format to temp file

plt.savefig(savedir+'/'+filename+'_massloss_temp.tiff',dpi=300,
                bbox_inches='tight')
# Covert uncompressed image to LZW compressed image
im = Image.open(savedir+'/'+filename+'_massloss_temp.tiff')
im.save(savedir+'/'+filename+'_massloss.tiff',compression='tiff_lzw')
# Close figure objects
plt.close()
im.close()
# Delete temp file
remove(savedir+'/'+filename+'_massloss_temp.tiff')

## Create figure visualizing extracted nonstoichiometry values vs. temperature
# Create figure object
plt.figure()
# Label axes
plt.xlabel('Temperature ($^\circ$C)')
if int_ref_flag == True:
    plt.ylabel('Nonstoichiometry, $\delta - \delta_0$')
else:
    plt.ylabel('Nonstoichiometry, $\delta$')
# Invert y axis
plt.gca().invert_yaxis()
# Plot nonstoichiometry vs temperature with error bars determined from error
# analysis
plt.errorbar(nonstoich_temp[:,0],nonstoich_temp[:,1],yerr=nonstoich_temp[:,2],
             fmt='ko', elinewidth=0.5,capsize=2, capthick=0.5)
# Save figure as tiff format to temp file
plt.savefig(savedir+'/'+filename+'_nonstoich_temp.tiff',dpi=300,
            bbox_inches='tight')
# Covert uncompressed image to LZW compressed image
im = Image.open(savedir+'/'+filename+'_nonstoich_temp.tiff')
im.save(savedir+'/'+filename+'_nonstoich.tiff',compression='tiff_lzw')
# Close figure objects
plt.close()
im.close()
# Delete temp file
remove(savedir+'/'+filename+'_nonstoich_temp.tiff')