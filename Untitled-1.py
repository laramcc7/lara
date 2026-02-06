# %%
import matplotlib.pyplot as plt        
import numpy as np                      
import pandas as pd                    
from scipy.optimize import curve_fit   
import scipy.optimize as op
from scipy.signal import find_peaks

data = []
with open('gammalong.txt') as f:
    for line in f:
        line = line.strip()
        if line.isdigit():
            data.append(int(line))
arr = np.array(data)



record_length = 256
waveforms = len(data) // record_length


Pulses = arr.reshape(waveforms, record_length)


#250 million samples per second, sorting x axis

samples = 250*10**6
time = 1/samples * 1*10**(9)


time_axis = np.arange(record_length) * time

#converting y axis 

adc_levels = 2**12       
vpp = 2.0                      
midpoint = adc_levels / 2      
Pulses_volts = (Pulses.astype(np.float32) - midpoint) * (vpp / adc_levels)


baseline_samples_start = 30  
baseline_samples_end = 80   

start_region = Pulses_volts[:, :baseline_samples_start]
end_region   = Pulses_volts[:, -baseline_samples_end:]

baseline_vals = np.mean(
    np.concatenate((start_region, end_region), axis=1),
    axis=1
)

Pulses_corrected = Pulses_volts - baseline_vals[:, np.newaxis]



# %%
plt.figure(figsize=(10,5))

idc = 12

plt.plot(Pulses_corrected[idc])
plt.xlabel("Time (ns)", size = 18)
#plt.xlim(350, 1100)
plt.title('Gamma Waveform')
plt.grid(True)
plt.show()

# %% [markdown]
# # FIRST AMPLITUDE SPECTRUMS

# %%
plt.figure(figsize=(16,5))


plt.subplot(1, 2, 1)
pulse_amplitudes = np.min(Pulses_corrected, axis=1)
plt.hist(pulse_amplitudes, bins=300)
plt.xlabel("Pulse Amplitude (V)")
plt.ylabel("Counts")
plt.title("Pulse Height Spectrum – Gamma Detector")
plt.yscale("log")
plt.grid(True)


plt.subplot(1, 2, 2)
plt.hist(pulse_amplitudes2, bins=300)
plt.xlabel("Pulse Amplitude (V)")
plt.ylabel("Counts")
plt.title("Pulse Height Spectrum – Alpha Detector")
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()


# %% [markdown]
# # SECOND AMPLITUDE SPECTRUMS

# %%
plt.figure(figsize=(16,5))
#cut
#gamma snr plot

plt.subplot(1, 2, 1)


baseline_region_1 = np.concatenate((Pulses_corrected[:, :baseline_samples_start], 
                                     Pulses_corrected[:, -baseline_samples_end:]), axis=1)


noiserms = np.std(baseline_region_1, axis=1)

snr1 = np.abs(pulse_amplitudes) / noiserms

valid = pulse_amplitudes < -3 * noiserms
valid_pulses = pulse_amplitudes[valid]

plt.hist(valid_pulses, bins=300)
plt.xlabel("Pulse Amplitude (V)")
plt.ylabel("Counts")
plt.title(f"Pulse amplitude spectrum - gamma")
plt.yscale("log")
plt.grid(True)


#alpha snr plot


plt.subplot(1, 2, 2)


baseline_region2 = np.concatenate((Pulses_corrected2[:, :baseline_samples_start], 
                                    Pulses_corrected2[:, -baseline_samples_end:]), axis=1)
noiserms2 = np.std(baseline_region2, axis=1)
snr2 = np.abs(pulse_amplitudes2) / noiserms2
snrthres2 = snr2 > 3
valid_pulses2 = pulse_amplitudes2[snrthres2]

plt.hist(valid_pulses2, bins=300)
plt.xlabel("Pulse Amplitude (V)")
plt.ylabel("Counts")
plt.title(f"Pulse amplitude spectrum - alpha")
plt.yscale("log")
plt.grid(True)


fixedthres = np.mean(-noiserms2) 
print(fixedthres)

plt.tight_layout()
plt.show()

# %% [markdown]
# 

# %%

gamma_thr = -0.15
alpha_thr = -0.20

gammatriggered = pulse_amplitudes < gamma_thr
alphatriggered = pulse_amplitudes2 < alpha_thr

plt.figure(figsize=(16,5))

#gamma spec when alpha fired
plt.subplot(1, 2, 1)

plt.hist(
    pulse_amplitudes[gammatriggered],
    bins=300,
    alpha=0.4,
    label="Gamma Amplitude Spectrum with threshold< -0.15v"
)

plt.hist(
    pulse_amplitudes[alphatriggered],
    bins=300,
    alpha=0.7,
    label="Gamma when alpha fired thres<-0.2v"
)

plt.xlabel("Gamma pulse amplitude (V)")
plt.ylabel("Counts")
plt.title('Gamma Pulse height when alpha fired')
plt.yscale("log")
plt.legend()
plt.grid(True)

#alpha pulse height when gam fired

plt.subplot(1, 2, 2)

plt.hist(
    pulse_amplitudes2[alphatriggered],
    bins=300,
    alpha=0.4,
    label="Alpha amp. spectrum thres < -0.2 v"
)

plt.hist(
    pulse_amplitudes2[gammatriggered],
    bins=300,
    alpha=0.7,
    label="Alpha when gamma fired thres < 0.15v"
)

plt.xlabel("Alpha pulse amplitude (V)")
plt.ylabel("Counts")
plt.title('Alpha Pulse height when gamma fired')
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# coindice timing with and without conditions

# %% [markdown]
# WITH:

# %%
valid_gamma = []
valid_gamma_time = []

for i, waveform in enumerate(Pulses_corrected):

    peaks, _ = find_peaks(-waveform, height=0.15, distance=25)

    if len(peaks) > 0:
        peak_sample = peaks[np.argmax(_["peak_heights"])]
        peak_time = time_axis[peak_sample]

        valid_gamma.append(i)
        valid_gamma_time.append(peak_time)

print(valid_gamma_time)

# %%
data2 = []
with open('alphalong.txt') as f:
    for line in f:
        line = line.strip()
        if line.isdigit():
            data2.append(int(line))
arr2 = np.array(data2)

print("numbers in text files", len(data2))

record_length = 256  
waveforms = len(data2) // record_length
print('number of waveforms in text file = ',waveforms)

Pulses2 = arr2.reshape(waveforms, record_length)


waveform = Pulses2[0]

#250 million samples per second

samples = 250*10**6
time = 1/samples * 1*10**(9)


time_axis = np.arange(record_length) * time

adc_levels = 2**12       
vpp = 2.0                      
midpoint = adc_levels / 2      
Pulses_volts2 = (Pulses2.astype(np.float32) - midpoint) * (vpp / adc_levels)
#do start and end
baseline_samples_start = 30  # 0-120 ns
baseline_samples_end = 100   # last 400 ns

start_region2 = Pulses_volts2[:, :baseline_samples_start]
end_region2   = Pulses_volts2[:, -baseline_samples_end:]

baseline_vals2 = np.mean(
    np.concatenate((start_region2, end_region2), axis=1),
    axis=1
)

Pulses_corrected2 = Pulses_volts2 - baseline_vals2[:, np.newaxis]


ind=3
plt.figure(figsize=(10,5))
plt.plot(time_axis, Pulses_corrected2[ind])
plt.xlabel("Time (ns)", size = 18)
#plt.xlim(1600, 2500)
plt.title('Alpha Waveform')
plt.grid(True)
plt.show()

# %%
ind=30
plt.figure(figsize=(10,5))
plt.plot(time_axis, Pulses_corrected2[ind])
plt.xlabel("Time (ns)", size = 18)
#plt.xlim(1600, 2500)
plt.title('Alpha Waveform')
plt.grid(True)
plt.show()

# %%
valid_alpha = []
valid_alpha_time = []

for i, waveform in enumerate(Pulses_corrected2):
    peaks, _ = find_peaks(-waveform, height=0.2, distance=50)

    if len(peaks) > 0:
        peak_sample = peaks[np.argmax(_["peak_heights"])]              
        peak_time = time_axis[peak_sample]   

        valid_alpha.append(i)          
        valid_alpha_time.append(peak_time)   


print(valid_alpha_time)

# %%
alpha_time = dict(zip(valid_alpha, valid_alpha_time))
gamma_time = dict(zip(valid_gamma, valid_gamma_time))

window_ns = 300 

coincidences = []
dt_values = []

for i in range(len(Pulses_corrected)):
    if i in alpha_time and i in gamma_time:

        dt = gamma_time[i] - alpha_time[i]   

        if 0 < dt < window_ns:               
            dt_values.append(dt)
            coincidences.append(i)

print("Coincident waveforms:", coincidences)

print(len(dt_values))


# %% [markdown]
# # TIME DIFFERENCE PLOTS

# %%
gamma_time = []
alpha_time = []

for waveform in Pulses_corrected:
    peak_sample = np.argmin(waveform)  
    gamma_time.append(time_axis[peak_sample])

for waveform in Pulses_corrected2:
    peak_sample = np.argmin(waveform)
    alpha_time.append(time_axis[peak_sample])

dt_valuesnc = []

for i in range(min(len(gamma_time), len(alpha_time))):
    dt = gamma_time[i] - alpha_time[i]
    dt_valuesnc.append(dt)

plt.figure()
plt.hist(dt_valuesnc, bins=100)
plt.xlabel("Δt = tγ − tα (ns)")
plt.ylabel("Counts")
plt.title("Timing difference histogram - Without conditions")
plt.grid()
plt.show()

# %%
alpha_time = dict(zip(valid_alpha, valid_alpha_time))
gamma_time = dict(zip(valid_gamma, valid_gamma_time))

window_ns = 300

dt_values = []

for i in alpha_time:
    if i in gamma_time:
        dt = gamma_time[i] - alpha_time[i]

        if 0 < dt < window_ns:
            dt_values.append(dt)

dt_values = np.array(dt_values)

print("Number of coincidences:", len(dt_values))

# %%
bins = 80
counts, bin_edges = np.histogram(dt_values, bins=bins)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

def exp_bg(t, A, tau, bg):
    return A * np.exp(-t / tau) + bg

t_min_fit = 12
t_max_fit = 300

mask = (
    (bin_centres > t_min_fit) &
    (bin_centres < t_max_fit) &
    (counts > 0)
)

x_fit = bin_centres[mask]
y_fit = counts[mask]
#y_err= np.sqrt(counts)

p0 = (y_fit.max(), 100, y_fit.min())

popt, pcov = curve_fit(exp_bg, x_fit, y_fit, p0=p0) #pass errors through data points, 

A_fit, tau_fit, bg_fit = popt
tau_err = np.sqrt(pcov[1, 1])

print(f"τ = {tau_fit:.2f} ± {tau_err:.2f} ns")

plt.figure(figsize=(8,5))

plt.bar(
    bin_centres,
    counts,
    width=bin_edges[1] - bin_edges[0],
    alpha=0.6,
    edgecolor="black"
)

t_plot = np.linspace(t_min_fit, t_max_fit, 400)

plt.plot(
    t_plot,
    exp_bg(t_plot, A_fit, tau_fit, bg_fit),
    'r',
    linewidth=2,
    label=f"Fit: τ = {tau_fit:.1f} ns"
)

plt.xlabel("Δt = tγ − tα (ns)")
plt.ylabel("Counts")
plt.title("Alpha–gamma coincidence timing")
plt.legend()
plt.grid(alpha=0.3)
plt.show()



