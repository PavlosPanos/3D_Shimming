import numpy as np
from shimming_3D import least_square_shimm_ROI
from shimming_3D import least_square_shim_masked
from shimming_info_retriever_3D import linear_coil_setings
from shimming_info_retriever_3D import quadradic_coil_setings

paths = [

    # OLD SCANS

    #"C:/Users/pavlo/Desktop/loeschen/010_ssfp_3D_cycl8_tune_up",
    #"C:/Users/pavlo/Desktop/loeschen/011_ssfp_3D_cycl8_tune_up",
    #"C:/Users/pavlo/Desktop/loeschen/012_ssfp_3D_cycl8_tune_up_p100Hz",
    #"C:/Users/pavlo/Desktop/loeschen/013_ssfp_3D_cycl8_tune_up_p100Hz",
    #"C:/Users/pavlo/Desktop/loeschen/014_ssfp_3D_cycl8_tune_up_p100A11",
    #"C:/Users/pavlo/Desktop/loeschen/015_ssfp_3D_cycl8_tune_up_p100A11",
    #"C:/Users/pavlo/Desktop/loeschen/016_ssfp_3D_cycl8_tune_up_p100B11",
    #"C:/Users/pavlo/Desktop/loeschen/017_ssfp_3D_cycl8_tune_up_p100B11",

    # New Scans

    "C:/Users/pavlo/Desktop/20240802/loeschen/050_ssfp_3D_cycl8_cor_tuneup",
    "C:/Users/pavlo/Desktop/20240802/loeschen/051_ssfp_3D_cycl8_cor_tuneup",

    # x-y-z Shimming Currents
    #"C:/Users/pavlo/Desktop/20240802/loeschen/052_ssfp_3D_cycl8_cor_A11_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/053_ssfp_3D_cycl8_cor_A11_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/054_ssfp_3D_cycl8_cor_B11_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/055_ssfp_3D_cycl8_cor_B11_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/056_ssfp_3D_cycl8_cor_A10_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/057_ssfp_3D_cycl8_cor_A10_p100",

    # 2nd order currents 
    #"C:/Users/pavlo/Desktop/20240802/loeschen/058_ssfp_3D_cycl8_cor_A20_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/059_ssfp_3D_cycl8_cor_A20_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/060_ssfp_3D_cycl8_cor_A21_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/061_ssfp_3D_cycl8_cor_A21_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/062_ssfp_3D_cycl8_cor_B21_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/063_ssfp_3D_cycl8_cor_B21_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/064_ssfp_3D_cycl8_cor_A22_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/065_ssfp_3D_cycl8_cor_A22_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/066_ssfp_3D_cycl8_cor_B22_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/067_ssfp_3D_cycl8_cor_B22_p100",
    
    
    #"C:/Users/pavlo/Desktop/20240802/loeschen/068_ssfp_3D_cycl8_cor_A11_m100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/069_ssfp_3D_cycl8_cor_A11_m100",

    #"C:/Users/pavlo/Desktop/20240802/loeschen/072_ssfp_3D_cycl8_cor_A00_p100",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/073_ssfp_3D_cycl8_cor_A00_p100"

    #"C:/Users/pavlo/Desktop/20240802/loeschen/074_ssfp_3D_cycl8_cor_A11_p100_4mm",
    #"C:/Users/pavlo/Desktop/20240802/loeschen/075_ssfp_3D_cycl8_cor_A11_p100_4mm",


]

# False -> Don't Produce the Maps --- True -> Create the interactive Maps

a,b,c,d,e,f,g,h,j = least_square_shim_masked(paths, plotting=True)

# ROI is by default 30X20X30 pixels around the isocentre (X*Y*Z)

#a,b,c,d,e,f,g,h,j = least_square_shimm_ROI(paths, plotting=True)


A00_set, A11_set, B11_set, A10_set = linear_coil_setings(paths[0])

A00_optimal = int(A00_set - a)
A11_optimal = np.round(A11_set - b,2)
B11_optimal = np.round(B11_set - c,2)
A10_optimal = np.round(A10_set - d,2)

A20_set, A21_set, B21_set, A22_set, B22_set = quadradic_coil_setings(paths[0])

A20_optimal = np.round(A20_set - e,2)
A21_optimal = np.round(A21_set - f,2)
B21_optimal = np.round(B21_set - g,2)
A22_optimal = np.round(A22_set - h,2)
B22_optimal = np.round(B22_set - j,2)


print(f"Set Frequency Value : A00 = {A00_set}")
print(f"Set Linear Shimming Current Values : A11 = {A11_set} , B11 = {B11_set} , A10 = {A10_set}")
print(f"Set Quadratic Shimming Current Values : A20 = {A20_set} , A21 = {A21_set} , B21 = {B21_set} , A22 = {A22_set} , B22 = {B22_set}")
print("-----------------------------------------")

print(f"Optimal Frequency Value : A00 = {A00_optimal}")
print(f"Optimal Linear Shimming Current Values : A11 = {A11_optimal} , B11 = {B11_optimal} , A10 = {A10_optimal}")
print(f"Optimal Quadratic Shimming Current Values : A20 = {A20_optimal} , A21 = {A21_optimal} , B21 = {B21_optimal} , A22 = {A22_optimal} , B22 = {B22_optimal}")
print("-----------------------------------------")
