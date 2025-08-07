Implementing methods of acoustic characterization (i.e. finding speed of sound and attenuation) in test tank. 
Based off methods introducted in "Improved ultrasonic spectroscopy methods for characterization of dispersive materials" by Haifeng Wang and Wenwu Cao
Created by Susan Burrell, R&D intern

METHOD 1: 
Relies on typical transmission spectroscopy methods, you need to accurately know the thickness of the material for calculations. 
May be difficult for soft or non-uniform materials. Code version is working properly. 

METHOD 2: 
Builds upon first method, but uses internal reflection to backsolve for material thickness. Better for applications 
with the material cases above, but difficult to resolve reflection from main transmission. Current code version is 
work in progress. May need implementation of different windowing methods or filtering for increased SNR.
