import os
import numpy as np
import warnings
import re


warnings.filterwarnings("ignore")

def linear_coil_setings(dicom_folder):
    
    def read_and_extract_metadata(dicom_path):
        if not os.path.isfile(dicom_path):
            raise ValueError("Provided path is not a file.")

        # Read the DICOM file as binary
        with open(dicom_path, 'rb') as f:
            binary_data = f.read()

        # Convert binary data to text
        try:
            text_data = binary_data.decode('ascii', errors='ignore')
        except UnicodeDecodeError:
            text_data = binary_data.decode('utf-8', errors='ignore')

        # Define regex patterns to find the metadata fields
        patterns = {
            'flSensitivityX': r'sGRADSPEC\.asGPAData\[0\]\.flSensitivityX\s*=\s*([0-9.]+)',
            'flSensitivityY': r'sGRADSPEC\.asGPAData\[0\]\.flSensitivityY\s*=\s*([0-9.]+)',
            'flSensitivityZ': r'sGRADSPEC\.asGPAData\[0\]\.flSensitivityZ\s*=\s*([0-9.]+)',
            'lOffsetX': r'sGRADSPEC\.asGPAData\[0\]\.lOffsetX\s*=\s*(-?\d+)',
            'lOffsetY': r'sGRADSPEC\.asGPAData\[0\]\.lOffsetY\s*=\s*(-?\d+)',
            'lOffsetZ': r'sGRADSPEC\.asGPAData\[0\]\.lOffsetZ\s*=\s*(-?\d+)',
            'lFrequency': r'sTXSPEC\.asNucleusInfo\[0\]\.lFrequency\s*=\s*([0-9.]+)'
        }

        # Search for patterns and extract values
        metadata = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text_data)
            if match:
                metadata[key] = match.group(1)
            else:
                metadata[key] = 'Not found'

        return metadata

    path = dicom_folder+"/image0001.dcm"
    metadata = read_and_extract_metadata(path)

    # Print the extracted metadata
    metadata_array = []
    for value in metadata.items():
        metadata_array.append(float(value[1]))


    A00_set = int(metadata_array[6])
    A11_set = np.round(metadata_array[3] * 1000 * metadata_array[0],2)
    B11_set = np.round(metadata_array[4] * 1000 * metadata_array[1],2)
    A10_set = np.round(metadata_array[5] * 1000 * metadata_array[2],2)

    return(A00_set, A11_set, B11_set, A10_set)

def quadradic_coil_setings(dicom_folder):
    
    def read_and_extract_metadata(dicom_path):
        if not os.path.isfile(dicom_path):
            raise ValueError("Provided path is not a file.")

        # Read the DICOM file as binary
        with open(dicom_path, 'rb') as f:
            binary_data = f.read()

        # Convert binary data to text
        try:
            text_data = binary_data.decode('ascii', errors='ignore')
        except UnicodeDecodeError:
            text_data = binary_data.decode('utf-8', errors='ignore')

        # Define regex patterns to find the shim current fields
        shim_patterns = {
            'alShimCurrent[0]': r'sGRADSPEC\.alShimCurrent\[0\]\s*=\s*(-?\d+)',
            'alShimCurrent[1]': r'sGRADSPEC\.alShimCurrent\[1\]\s*=\s*(-?\d+)',
            'alShimCurrent[2]': r'sGRADSPEC\.alShimCurrent\[2\]\s*=\s*(-?\d+)',
            'alShimCurrent[3]': r'sGRADSPEC\.alShimCurrent\[3\]\s*=\s*(-?\d+)',
            'alShimCurrent[4]': r'sGRADSPEC\.alShimCurrent\[4\]\s*=\s*(-?\d+)'
        }

        # Search for patterns and extract values
        shim_metadata = {}
        for key, pattern in shim_patterns.items():
            match = re.search(pattern, text_data)
            if match:
                shim_metadata[key] = match.group(1)
            else:
                shim_metadata[key] = 'Not found'

        # Convert shim metadata dictionary to a list of tuples
        shim_metadata_list = [(key, value) for key, value in shim_metadata.items()]

        return shim_metadata_list

    path = dicom_folder+"/image0001.dcm"
    metadata = read_and_extract_metadata(path)

    # Print the extracted metadata
    metadata_array = []
    for value in metadata:
        metadata_array.append(float(value[1]))

    alShimCurrent_A20 = metadata_array[0]
    alShimCurrent_A21 = metadata_array[1]
    alShimCurrent_B21 = metadata_array[2]
    alShimCurrent_A22 = metadata_array[3]
    alShimCurrent_B22 = metadata_array[4]


    # THESE SHOULD CHANGE
    ShimSens_0 = 31.0
    ShimSens_1 = 22.2
    ShimSens_2 = 21.9
    ShimSens_3 = 22.2
    ShimSens_4 = 21.8

    ShimRefRadius = 0.25

    A20_set = np.round(alShimCurrent_A20 * ShimSens_0 * (ShimRefRadius**2),2)
    A21_set = np.round(alShimCurrent_A21 * ShimSens_1 * (ShimRefRadius**2),2)
    B21_set = np.round(alShimCurrent_B21 * ShimSens_2 * (ShimRefRadius**2),2)
    A22_set = np.round(alShimCurrent_A22 * ShimSens_3 * (ShimRefRadius**2),2)
    B22_set = np.round(alShimCurrent_B22 * ShimSens_4 * (ShimRefRadius**2),2)


    return(A20_set, A21_set, B21_set, A22_set, B22_set)



