Implementing methods of acoustic characterization (i.e. finding speed of sound and attenuation) in test tank. 
Based off methods introducted in "Improved ultrasonic spectroscopy methods for characterization of dispersive materials" by Haifeng Wang and Wenwu Cao.

Use fft_plotting script to read in files individually and check if needed. 

METHOD 1: 
Relies on typical transmission spectroscopy methods, you need to accurately know the thickness of the material for calculations. 
May be difficult for soft or non-uniform materials. Code version is working properly. 

Input variables: speed of sound in water, density of water, density of sample, thickness of sample, center frequency of signal.

METHOD 2: 
Builds upon first method, but uses internal reflection to backsolve for material thickness. Better for applications 
with the material cases above, but difficult to resolve reflection from main transmission. Current code version is 
work in progress. May need implementation of different windowing methods or filtering for increased SNR. Method 2 
requires separation of main transmission from reflection, alter windowing parameters and center time accordingly. Visually inspect
using plots. Once determined hold same for each sample and water signal pair. Higher number of input waveforms the better to
eliminate variability, raw signal averaging function included in script.

Input variables: speed of sound in water, density of water, density of sample, center frequency of signal. 

When using test tank, take new reference water signals for each new sample tested. Hold all variables the same, 
movement / misalignment of probe can introduce errors. Use triggered oscope for waveform input (keep trig level same).
Use function generator in single burst mode for sine waves, increase Vpp if RF amplifier is not used to raise 
signal above noise floor. 
