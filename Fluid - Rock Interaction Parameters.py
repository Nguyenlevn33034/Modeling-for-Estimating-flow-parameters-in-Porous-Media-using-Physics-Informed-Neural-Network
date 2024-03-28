'''
Tugas Besar Buckley Leverett Simulator EOR
Kelompok A14
Nama Anggota Kelompok:
1. Theresia Bonita Elsa Manora (101320049)
2. Nicholin Anggel Wairissal (101320051)
3. Justu Imanuel Izaac (101320079)
4. Nurpika Adilla (101320093)
'''

# HOW TO USE VIDEO LINK
'''https://youtu.be/z2Er7qYDLDI'''

# IMPORT LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# INPUT
## Relative Permeability
S_wc = input('Water Connate Saturation: ')
S_or = input('Oil Residual Saturation: ')
k_rw_S_or = input('Water Relative Permeability at Water Connate Saturation (mD): ')
k_ro_S_wc = input('Oil Relative Permeability at Oil Residual Saturation (mD): ')
n_w = input('Exponent Water (1-6): ')
n_o = input('Exponent Oil (1-6): ')

## Reservoir Properties
porosity = input('Porosity: ')
width = input('Width (ft): ')
length = input('Length (ft): ')
thick = input('Thickness (ft): ')

## Fluid Properties
visco_o = input('Oil Viscosity (cp): ')
visco_w = input('Water Viscosity (cp): ')

## Injection Rate
q_inj = input('Injection rate (bbl/d): ')


# PROCESS
## Convert String to Float
S_or = float(S_or)
S_wc = float(S_wc)
k_ro_S_wc = float(k_ro_S_wc)
k_rw_S_or = float(k_rw_S_or)
n_w = float(n_w)
n_o = float(n_o)
porosity = float(porosity)
width = float(width)
length = float(length)
thick = float(thick)
visco_o = float(visco_o)
visco_w = float(visco_w)
q_inj = float(q_inj)

## Setup
### Fractional Flow Setup
n = 100 # Total Iteration

S_w_list = [] # Water Saturation List
k_rw_list = [] # Water Relative Permeability List
k_ro_list = [] # Oil Relative Permeability List
f_w_list = [] # Fractional Flow List
df_w_per_dS_w_list = [] # df_w/dS_w List
slope_list = [] # Slope Between Two Point List
diff_list = [] # Difference Between df_w/dS_w and Slope List

S_w = S_wc # Initial Water Saturation

### Production Profile Setup
period_list = ['Initial', 'Before Breakthrough', 'At Breakthrough', 'After Breakthrough 1', 'After Breakthrough 2', 'After Breakthrough 3'] # Period List
S_wf_list = [] # Water Saturation at Front List
S_w_avg_list = [] # Average Water Saturation List
f_wf_list = [] # Fractional Flow at Front List
t_list = [] # Time List
pv_list = [] # Injected Water Volume (PV) List
q_o_list = [] # Oil Production Rate List
cum_prod_list = [] # Cumulative Oil Production List
wc_list = [] # Water Cut List
wor_list = [] # Water Oil Ratio List
rf_list = [] # Recovery Factor List

## Fractional Flow Calculation
for i in range(n):
    ### Water Saturation Step
    S_w_list.append(S_w)
    step = ((1-S_or-S_wc)/(n-1))
    S_wD = (S_w-S_wc)/(1-S_or-S_wc)
    S_w += step

    ### Water Relative Permeability Calculation
    k_rw = k_rw_S_or*(S_wD**n_w)
    k_rw_list.append(k_rw)

    ### Oil Relative Permeability Calculation
    k_ro = k_ro_S_wc*((1-S_wD)**n_o)
    k_ro_list.append(k_ro)
    
    ### Fractional Flow Calculation
    if i == 0:
        f_w = 0
    else:
        f_w = 1/(1+(k_ro/visco_o)*(visco_w/k_rw))
    f_w_list.append(f_w)

    ### df_w/dS_w Calculation
    if i > 0:
        df_w_per_dS_w = (f_w)/(S_w-S_wc)
    else:
        df_w_per_dS_w = '-'
    df_w_per_dS_w_list.append(df_w_per_dS_w)

for i in range(n):
    ### Slope Beetween Two Point Calculation
    if i > 0:
        if i == n-1:
            slope = (f_w_list[i]-f_w_list[i-1])/(S_w_list[i]-S_w_list[i-1])
        else:
            slope = (f_w_list[i+1]-f_w_list[i-1])/(S_w_list[i+1]-S_w_list[i-1])

        ### Absolute Difference Between df_w/dS_w and Slope
        diff = abs(df_w_per_dS_w_list[i]-slope)
    else:
        slope = '-'
        diff = '-'

    diff_list.append(diff)
    slope_list.append(slope)
    
## Determine Breakthrough
### Find Smallest Difference Value
diff_bt = diff_list[int(0.3*n)] # Assume breakthrough happen after 30% of Waterflooding Process
for i in range(n):
    if i > 0.3*n and diff_list[i] < diff_bt:
        diff_bt = diff_list[i]
        n_bt = i # n-Iteration at breakthrough

## Production Profile Calculation
A = width*thick
for i in range(len(period_list)):
    ### Initial Period
    if 'initial' in period_list[i].lower():
        S_wf = '-' # Water Saturation at Front
        f_wf = '-' # Fractional Flow at Front
        S_w_avg = '-' # Average Water Saturatioin
        t = 0 # Time
        pv = 0 # Injected Water Volume (PV)
        q_o = q_inj # Oil Production Rate
        cum_prod = 0 # Cumulative Oil Production
        wc = 0 # Watercut
        wor = (wc/100)/(1-(wc/100)) # Water Oil Ratio
        rf = 0 # Recovery Factor

    ### Berfore Breakthrough Period
    elif 'before' in period_list[i].lower():
        S_wf = '-'
        f_wf = '-'
        S_w_avg = '-'
        t = 0
        pv = 0
        q_o = q_o_list[i-1]
        cum_prod = 0
        wc = 0
        wor = 0
        rf = 0

    ### At Breakthrough Period
    elif 'at' in period_list[i].lower():
        S_wf = S_w_list[n_bt]
        f_wf = f_w_list[n_bt]
        S_w_avg = ((S_wf-S_wc)/f_wf) + S_wc
        S_wf_bt = S_wf
        f_wf_bt = f_wf
        S_w_avg_bt = S_w_avg
        t = (A*length*porosity/5.615)/(q_inj)*(S_w_avg-S_wc)
        t_list[i-1] = t
        pv = S_w_avg-S_wc
        pv_list[i-1] = pv
        q_o = q_inj*(1-f_wf)
        cum_prod = (A*length*porosity/5.615)*(pv)
        cum_prod_list[i-1] = cum_prod
        wc = f_wf*100
        wc_list[i-1] = wc
        wor = (wc/100)/(1-(wc/100))
        wor_list[i-1] = wor
        rf = (S_w_avg-S_wc)/(1-S_wc)*100
        rf_list[i-1] = rf

    ### After Breakthrough Period
    else:
        S_wf = S_w_list[n_bt + (i-2)*int(0.2*(n-n_bt))]
        f_wf = f_w_list[n_bt + (i-2)*int(0.2*(n-n_bt))]
        S_w_avg = ((S_wf-S_wc)/f_wf) + S_wc
        t = (A*length*porosity/5.615)/(q_inj)*((S_w_avg-S_wf)/(1-f_wf))
        pv = (S_w_avg-S_wf)/(1-f_wf)
        q_o = q_inj*(1-f_wf)
        cum_prod = (A*porosity*length/5.615)*(pv)
        wc = f_wf*100
        wor = (wc/100)/(1-(wc/100))
        rf = (S_w_avg-S_wc)/(1-S_wc)*100

    S_wf_list.append(S_wf)
    S_w_avg_list.append(S_w_avg)
    f_wf_list.append(f_wf)
    t_list.append(t)
    pv_list.append(pv)
    q_o_list.append(q_o)
    cum_prod_list.append(cum_prod)
    wc_list.append(wc)
    wor_list.append(wor)
    rf_list.append(rf)

# OUTPUT
## Table
### Fractional Flow Table
print('Fractional Flow Table')
pd.set_option('display.max_rows', None)
d_1 = {'Sw': S_w_list, 'Krw (mD)': k_rw_list, 'Kro (mD)': k_ro_list, 'fw': f_w_list}
df_1 = pd.DataFrame(data=d_1)
print(df_1, '\n')

### Production Profile Table
print('Production Profile Table')
pd.set_option('display.max_columns', None)
d_2 = {'Period': period_list, 'Swf': S_wf_list, 'Sw avg': S_w_avg_list, 'fwf': f_wf_list, 'Time (d)': t_list, 'Injected Water Volume (PV)': pv_list, 'Oil Production Rate (bbl/d)': q_o_list, 'Cumulative Oil Production (bbl)': cum_prod_list, 'Watercut (%)': wc_list, 'Water Oil Ratio': wor_list, 'Recovery Factor (%)': rf_list}
df_2 = pd.DataFrame(data=d_2)
print(df_2, '\n')

## Plot Curve
### Relative Permeability Curve
fig1 = plt.figure(1, layout='constrained')
plt.plot(np.array(S_w_list), np.array(k_rw_list))
plt.plot(np.array(S_w_list), np.array(k_ro_list))
plt.scatter(np.array([S_wc, 1-S_or, S_wc, 1-S_or]), np.array([0, 0, k_ro_S_wc, k_rw_S_or]), color = '#999999')
plt.text(S_wc, 0, 'Swc')
plt.text(1-S_or, 0, '1-Sor')
plt.text(S_wc, k_ro_S_wc, 'Kro@Swc')
plt.text(1-S_or, k_rw_S_or, 'Krw@Sor')
plt.xlabel('Sw')
plt.ylabel('Kr')
plt.title('Relative Permeability Curve')

### Fractional Flow Curve
fig2 = plt.figure(2, layout='constrained')
plt.plot(np.array(S_w_list), np.array(f_w_list))
plt.plot(np.array([S_wc, S_w_avg_bt]), np.array([0, 1]))
plt.plot(np.array([S_wf_bt, S_wf_bt]), np.array([0, f_wf_bt]), '#999999')
plt.plot(np.array([S_wc, S_wf_bt]), np.array([f_wf_bt, f_wf_bt]), '#999999')
plt.plot(np.array([S_w_avg_bt, S_w_avg_bt]), np.array([0, 1]), '#999999')
plt.scatter(np.array([S_wf_bt, S_w_avg_bt, S_wc]), np.array([0, 0, f_wf_bt]), color = '#999999')
plt.text(S_wf_bt, 0, 'Swf')
plt.text(S_w_avg_bt, 0, 'Sw Avg')
plt.text(S_wc, f_wf_bt, 'fwf')
plt.xlabel('Sw')
plt.ylabel('fw')
plt.title('Fractional Flow Curve')

### Production Profile Curve
fig3 = plt.figure(3, layout='constrained')
fig3.suptitle('Production Profile')
#### Time vs Injected Water Volume (PV)
plot1 = plt.subplot(231)
plot1.plot(np.array(t_list), np.array(pv_list))
plot1.set_xlabel('Time (d)')
plot1.set_ylabel('Qi (PV)')

#### Time vs Oil Production Rate
plot2 = plt.subplot(232)
plot2.plot(np.array(t_list), np.array(q_o_list))
plot2.set_xlabel('Time (d)')
plot2.set_ylabel('Qo (bbl/d)')

#### Time vs Cumulative Oil Production
plot3 = plt.subplot(233)
plot3.plot(np.array(t_list), np.array(cum_prod_list))
plot3.set_xlabel('Time (d)')
plot3.set_ylabel('Np (bbl)')

#### Time vs Water Cut & Recovery Factor
plot4 = plt.subplot(234)
plot4.plot(np.array(t_list), np.array(wc_list))
plot4.plot(np.array(t_list), np.array(rf_list))
plot4.legend(['WC', 'RF'])
plot4.set_xlabel('Time (d)')
plot4.set_ylabel('WC & RF (%)')

#### Time vs Water Oil Ratio
plot5 = plt.subplot(235)
plot5.plot(np.array(t_list), np.array(wor_list))
plot5.set_xlabel('Time (d)')
plot5.set_ylabel('WOR')

### Display
plt.show()