import os
import numpy as np
import pydicom
from matplotlib.patches import Rectangle
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pydicom.fileset import FileSet
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

def load3d_dicom(path):
    if not os.path.isdir(path):
        raise ValueError("Provided path is not a directory.")

    # List all DICOM files in the directory
    dicom_files = [f for f in os.listdir(path) if f.endswith('.dcm') and not f.startswith('.')]

    # Check for DICOMDIR and use it if present
    if 'DICOMDIR' in dicom_files:
        dicomdir_path = os.path.join(path, 'DICOMDIR')
        fs = FileSet(dicomdir_path)
        dicom_files = [instance.path for instance in fs]
        print(f"Loaded DICOMDIR with {len(dicom_files)} instances.")
    else:
        dicom_files = [os.path.join(path, f) for f in dicom_files]

    # Sort DICOM files based on numerical values in filenames
    dicom_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.splitext(f)[0])) or '0'))

    if not dicom_files:
        raise ValueError("No DICOM files found in the directory.")

    # Read the first DICOM file to extract metadata
    ds = pydicom.dcmread(dicom_files[0])
    n_rows = ds.Rows
    n_cols = ds.Columns
    n_slices = len(dicom_files)
    
    pixel_spacing = ds.PixelSpacing if 'PixelSpacing' in ds else [1.0, 1.0]
    slice_thickness = ds.SliceThickness if 'SliceThickness' in ds else 1.0
    true_color = (ds.PhotometricInterpretation == 'RGB')
    
    if true_color:
        volume = np.zeros((n_rows, n_cols, n_slices, 3), dtype=np.float32)
    else:
        volume = np.zeros((n_rows, n_cols, n_slices), dtype=np.float32)
    
    info = []
    tr_time = None

    # Read each DICOM file and store pixel data
    for curslice, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        info.append(ds)
        if true_color:
            volume[:, :, curslice, :] = ds.pixel_array.astype(np.float32) / 255.0
        else:
            volume[:, :, curslice] = ds.pixel_array.astype(np.float32)
        
        if tr_time is None and 'RepetitionTime' in ds:
            tr_time = ds.RepetitionTime
    
    voxel_size = (pixel_spacing[0], pixel_spacing[1], slice_thickness)

    return volume, info, voxel_size, tr_time

def phaze_transformation(volume_phase):
    return (volume_phase - np.max(volume_phase) / 2) / np.max(volume_phase) * 2 * np.pi

def model(params, A00,  A11, B11, A10, A20, A21, B21, A22, B22):
    a, b, c, d, e, f, g, h, j = params
    return(a * A00 + b * A11 + c * B11 + d * A10 + e*A20 + f*A21 + g*B21 + h*A22 + j*B22)

def residuals(params, A00, A11, B11, A10, A20, A21, B21, A22, B22, optimal_result):
    try:
        model_values = model(params, A00, A11, B11, A10, A20, A21, B21, A22, B22)
        if np.any(np.isnan(model_values)) or np.any(np.isinf(model_values)):
            raise ValueError("Model function produced NaNs or Infs.")
        return model_values - optimal_result
    except Exception as e:
        print(f"Error in residuals function: {e}")
        raise

def shimming_base_function(coil_name, x_voxels, y_voxels, z_voxels):
    # Initialize the base function array with ones
    base_function = np.ones((len(x_voxels), len(y_voxels), len(z_voxels)))
    
    if coil_name == "A00":
        # For A00, the base function is simply ones
        return base_function
    
    elif coil_name == "A11":
        # For A11, fill with x_voxels along the x-axis
        for i in range(len(x_voxels)):
            for j in range(len(y_voxels)):
                base_function[i, :, j] *= x_voxels

        return base_function
    
    elif coil_name == "B11":
        # For B11, fill with y_voxels along the y-axis
        for i in range(len(x_voxels)):
            for j in range(len(z_voxels)):
                base_function[i, j, :] *= y_voxels
        return base_function
    
    elif coil_name == "A10":
        # For A10, fill with z_voxels along the z-axis
        for i in range(len(y_voxels)):
            for j in range(len(z_voxels)):
                base_function[:, j, i] *= z_voxels
        return base_function

    elif coil_name == "A20":
        for i in range(len(y_voxels)):
            for j in range(len(z_voxels)):
                base_function[:, j, i] *= z_voxels ** 2
        for i in range(len(x_voxels)):
            for j in range(len(z_voxels)):
                base_function[i, j, :] -= 0.5*y_voxels**2
        for i in range(len(x_voxels)):
            for j in range(len(y_voxels)):
                base_function[i, :, j] -= 0.5*x_voxels**2
        return base_function
    
    elif coil_name == "A21":
        for i in range(len(x_voxels)):
            for j in range(len(y_voxels)):
                base_function[i, :, j] *= 2*x_voxels
        for i in range(len(y_voxels)):
            for j in range(len(z_voxels)):
                base_function[:, j, i] *= z_voxels
        return base_function
    
    elif coil_name == "B21":
        for i in range(len(y_voxels)):
            for j in range(len(z_voxels)):
                base_function[:, j, i] *= 2*z_voxels
        for i in range(len(x_voxels)):
            for j in range(len(z_voxels)):
                base_function[i, j, :] *= y_voxels
        return base_function
    
    elif coil_name == "A22":
        for i in range(len(x_voxels)):
            for j in range(len(z_voxels)):
                base_function[i, j, :] = -y_voxels ** 2
        for i in range(len(x_voxels)):
            for j in range(len(y_voxels)):
                base_function[i, :, j] += x_voxels ** 2
        return base_function

    elif coil_name == "B22":
        for i in range(len(x_voxels)):
            for j in range(len(z_voxels)):
                base_function[i, j, :] *= 2*y_voxels
        for i in range(len(x_voxels)):
            for j in range(len(y_voxels)):
                base_function[i, :, j] *= x_voxels
        return base_function
        
    
    else:
        raise ValueError(f"Unknown coil name: {coil_name}")

def get_subvolume(volume, x_start, x_end, y_start, y_end, z_start, z_end):
    return volume[x_start:x_end,z_start:z_end ,y_start:y_end]


def least_square_shimm_ROI(paths,plotting):

    def FFT_Off_Resonance(complex_signal, TR):
        N = complex_signal.shape[-1]
        
        # Perform FFT and shift the zero-frequency component to the center
        fft_shifted = np.fft.fftshift(np.fft.fft(complex_signal, axis=-1) / N, axes=-1)
        
        # Calculate the phase difference
        center_index = N // 2
        phase_diff = np.angle(fft_shifted[..., center_index] / fft_shifted[..., center_index - 1])
        
        # Convert phase difference to degrees and compute Off-Resonance Map
        off_resonance_map = np.rad2deg(phase_diff) / TR * 1000 / 360
        
        return off_resonance_map


    def plot_interactive_heatmap_with_roi(Off_Resonance_Map, Off_Resonance_Map_A00, Off_Resonance_Map_A11, 
                                        Off_Resonance_Map_B11, Off_Resonance_Map_A10, Off_Resonance_Map_A20, 
                                        Off_Resonance_Map_A21, Off_Resonance_Map_B21, Off_Resonance_Map_A22, 
                                        Off_Resonance_Map_B22, a, b, c, d, e, f, g, h, j, roi_coords, voxel_size):
        x_start, x_end, y_start, y_end, z_start, z_end = roi_coords
        
        x_voxels = (np.arange(Off_Resonance_Map.shape[0]) * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[0] / 2)
        y_voxels = (np.arange(Off_Resonance_Map.shape[2]) * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2)
        z_voxels = (np.arange(Off_Resonance_Map.shape[1]) * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2)

        x_start_voxel = x_start * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[1] / 2
        x_end_voxel = x_end * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[1] / 2
        y_start_voxel = y_start * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2
        y_end_voxel = y_end * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2
        z_start_voxel = z_start * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2
        z_end_voxel = z_end * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, wspace=0.2)
        
        # Initialize the heatmaps with the middle slice
        heatmap1 = ax1.imshow(Off_Resonance_Map[:, :, int(Off_Resonance_Map.shape[2] / 2)],
                            cmap="viridis",extent=[x_voxels.min(), x_voxels.max(), z_voxels.min(), z_voxels.max()])
        plt.colorbar(heatmap1, ax=ax1)
        ax1.set_title("Off-Resonance 2-D Heat Map (slice)")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Z (mm)")

        vmin, vmax = heatmap1.get_clim()
        
        # Initialize the difference map
        diff_map = Off_Resonance_Map - (a*Off_Resonance_Map_A00+b*Off_Resonance_Map_A11
                                        +c*Off_Resonance_Map_B11+d*Off_Resonance_Map_A10
                                        +e*Off_Resonance_Map_A20+f*Off_Resonance_Map_A21
                                        +g*Off_Resonance_Map_B21+h*Off_Resonance_Map_A22
                                        +j*Off_Resonance_Map_B22)
        heatmap2 = ax2.imshow(diff_map[:, :, int(diff_map.shape[2] / 2)],
                            cmap="viridis", vmin=vmin, vmax=vmax,
                            extent=[x_voxels.min(), x_voxels.max(), z_voxels.min(), z_voxels.max()])
        plt.colorbar(heatmap2, ax=ax2)
        ax2.set_title("Shimmed Map (slice)")
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Z (mm)")
        
        # Histogram plot
        flat_non_corrected_roi = get_subvolume(Off_Resonance_Map, x_start, x_end, y_start, y_end, z_start, z_end)
        flat_non_corrected_roi = flat_non_corrected_roi.flatten()
        flat_corrected_roi = get_subvolume(diff_map, x_start, x_end, y_start, y_end, z_start, z_end)
        flat_corrected_roi = flat_corrected_roi.flatten()
        ax3.hist(flat_non_corrected_roi, bins=50, alpha=0.5, label='Raw')
        ax3.hist(flat_corrected_roi, bins=50, alpha=0.5, label='Shimmed')
        ax3.set_title("Histogram of Frequencies (ROI)")
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Count")
        ax3.legend()
        
        # Function to create a rectangle patch
        def create_rectangle(x_start, z_start, x_end, z_end):
            return Rectangle((x_start, z_start), x_end - x_start, z_end - z_start,
                            linewidth=2, edgecolor='r', facecolor='none', linestyle='--')

        def update(val):
            """Update function for the slice slider."""
            slice_index = int(slice_slider.val)
            heatmap1.set_data(Off_Resonance_Map[:, :, slice_index])
            diff_map_slice = diff_map[:, :, slice_index]
            heatmap2.set_data(diff_map_slice)
            
            # Remove all previous rectangles
            for patch in ax1.patches:
                patch.remove()
            for patch in ax2.patches:
                patch.remove()
            
            # Add rectangle if the current slice is within the ROI
            if y_start <= slice_index <= y_end:
                rect = create_rectangle(x_start_voxel, z_start_voxel, x_end_voxel, z_end_voxel)
                ax1.add_patch(rect)
                rect = create_rectangle(x_start_voxel, z_start_voxel, x_end_voxel, z_end_voxel)
                ax2.add_patch(rect)
            fig.canvas.draw_idle()
        
        # Slider for selecting the slice
        ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
        slice_slider = Slider(ax_slice, 'y-Slice', 0, Off_Resonance_Map.shape[2] - 1, 
                            valinit=int(Off_Resonance_Map.shape[2] / 2), valstep=1)
        slice_slider.on_changed(update)
        
        plt.show()

    N = 8
    # Load data from each path
    volumes = []
    infos = []
    voxel_sizes = []
    for path in paths:
        volume, info, voxel_size , tr_time = load3d_dicom(path)
        volumes.append(volume)
        infos.append(info)
        voxel_sizes.append(voxel_size)
    
    volume_magnitude, volume_phase = volumes[0], volumes[1]
    voxel_size = voxel_sizes[0]

    volume_phase = phaze_transformation(volume_phase)
    complex_magnitude = volume_magnitude * np.exp(-1j * volume_phase)
    four_dimension_magnitude_data = np.array(complex_magnitude).reshape(
        (volume_magnitude.shape[0],volume_magnitude.shape[1],
         int(volume_magnitude.shape[2]/N),N), order="F")
    Off_Resonance_Map = FFT_Off_Resonance(four_dimension_magnitude_data,tr_time)

    x_start, x_end = int(volume_magnitude.shape[0]/2 - 15), int(volume_magnitude.shape[0]/2 + 15)
    y_start, y_end = int(volume_magnitude.shape[0]/2 - 10), int(volume_magnitude.shape[0]/2 + 10)
    z_start, z_end = int(volume_magnitude.shape[0]/2 - 15), int(volume_magnitude.shape[0]/2 + 15)

    x_voxels = (np.arange(Off_Resonance_Map.shape[0]) * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[0] / 2)
    y_voxels = (np.arange(Off_Resonance_Map.shape[2]) * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2)
    z_voxels = (np.arange(Off_Resonance_Map.shape[1]) * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2)


    Off_Resonance_Map_A00 = shimming_base_function("A00",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A11 = shimming_base_function("A11",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B11 = shimming_base_function("B11",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A10 = shimming_base_function("A10",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A20 = shimming_base_function("A20",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A21 = shimming_base_function("A21",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B21 = shimming_base_function("B21",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A22 = shimming_base_function("A22",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B22 = shimming_base_function("B22",x_voxels,y_voxels,z_voxels)


    Off_Resonance_Map_roi = get_subvolume(Off_Resonance_Map, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A00_roi = get_subvolume(Off_Resonance_Map_A00, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A11_roi = get_subvolume(Off_Resonance_Map_A11, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_B11_roi = get_subvolume(Off_Resonance_Map_B11, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A10_roi = get_subvolume(Off_Resonance_Map_A10, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A20_roi = get_subvolume(Off_Resonance_Map_A20, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A21_roi = get_subvolume(Off_Resonance_Map_A21, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_B21_roi = get_subvolume(Off_Resonance_Map_B21, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_A22_roi = get_subvolume(Off_Resonance_Map_A22, x_start, x_end, y_start, y_end, z_start, z_end)
    Off_Resonance_Map_B22_roi = get_subvolume(Off_Resonance_Map_B22, x_start, x_end, y_start, y_end, z_start, z_end)


    A00_maps = Off_Resonance_Map_A00_roi
    A11_maps = Off_Resonance_Map_A11_roi
    B11_maps = Off_Resonance_Map_B11_roi
    A10_maps = Off_Resonance_Map_A10_roi
    A20_maps = Off_Resonance_Map_A20_roi
    A21_maps = Off_Resonance_Map_A21_roi
    B21_maps = Off_Resonance_Map_B21_roi
    A22_maps = Off_Resonance_Map_A22_roi
    B22_maps = Off_Resonance_Map_B22_roi
    data = Off_Resonance_Map_roi
    
    mask =  ~np.isnan(data) & ~np.isnan(A00_maps) & ~np.isnan(A11_maps) & ~np.isnan(B11_maps) & ~np.isnan(A10_maps) & ~np.isnan(A20_maps) & ~np.isnan(A21_maps) & ~np.isnan(B21_maps) & ~np.isnan(A22_maps) & ~np.isnan(B22_maps)
    A00_maps = A00_maps[mask]
    A11_maps = A11_maps[mask]
    B11_maps = B11_maps[mask]
    A10_maps = A10_maps[mask]
    A20_maps = A20_maps[mask]
    A21_maps = A21_maps[mask]
    B21_maps = B21_maps[mask]
    A22_maps = A22_maps[mask]
    B22_maps = B22_maps[mask]
    data = data[mask]

    initial_guess = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    result = least_squares(residuals, initial_guess, loss='soft_l1', args=(A00_maps, A11_maps, B11_maps, A10_maps, A20_maps, A21_maps, B21_maps, A22_maps, B22_maps, data))

    """
                        --- Least Square Possible Loss Funtions ---

    linear (default) : rho(z) = z. Gives a standard least-squares problem.
    soft_l1 : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
    huber : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to soft_l1.
    cauchy : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.
    arctan : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to cauchy.
    """

    a, b, c, d, e, f, g, h, j = result.x

    #print(f"Optimal Shimming Values Change Relative to tune-up: A00 = {np.round(a,2)}, A11 = {-np.round(b*10**3/42.58,2)}, B11 = {np.round(c*10**3/42.58,2)}, A10 = {-np.round(d*10**3/42.58,2)}")
    #print(f"Optimal Shimming Values Change Relative to tune-up: A20 = {-np.round(e*10**6/42.58,2)} , A21 = {-np.round(f*10**6/42.58,2)} , B21 = {np.round(g*10**6/42.58,2)} , A22 = {-np.round(h*10**6/42.58,2)} , B22 = {np.round(j*10**6/42.58,2)}")

    if plotting == True :
        plot_interactive_heatmap_with_roi(Off_Resonance_Map, Off_Resonance_Map_A00, Off_Resonance_Map_A11, Off_Resonance_Map_B11,Off_Resonance_Map_A10, Off_Resonance_Map_A20, Off_Resonance_Map_A21, Off_Resonance_Map_B21, Off_Resonance_Map_A22, Off_Resonance_Map_B22,
                                        a, b, c, d, e, f, g, h, j, (x_start, x_end, y_start, y_end, z_start, z_end), voxel_size)
        
    # The parameters we want to actually retrieve 
    a = np.round(a)
    b = -np.round(b*10**3/42.58,2)
    c = np.round(c*10**3/42.58,2)
    d = -np.round(d*10**3/42.58,2)
    e = -np.round(e*10**6/42.58,2) 
    f = -np.round(f*10**6/42.58,2)
    g = np.round(g*10**6/42.58,2)
    h = -np.round(h*10**6/42.58,2)
    j = np.round(j*10**6/42.58,2)

    # I need to make this script as a funtion that is called by another script and returns the differences rellative to tuna-up setings


    return(a,b,c,d,e,f,g,h,j)

def least_square_shim_masked(paths,plotting):

    def FFT_Off_Resonance_Threshold(complex_signal, TR):
        N = complex_signal.shape[-1]
        
        # Perform FFT and shift the zero-frequency component to the center
        fft_shifted = np.fft.fftshift(np.fft.fft(complex_signal, axis=-1) / N, axes=-1)
        
        # Calculate the phase difference
        center_index = N // 2
        phase_diff = np.angle(fft_shifted[..., center_index] / fft_shifted[..., center_index - 1])
        
        # Convert phase difference to degrees and compute Off-Resonance Map
        off_resonance_map = np.rad2deg(phase_diff) / TR * 1000 / 360
        zeroth_order_magnitude = np.abs(fft_shifted[..., center_index])

        
        return (off_resonance_map,zeroth_order_magnitude)

    def plot_interactive_heatmap_with_roi(Off_Resonance_Map, Off_Resonance_Map_A00, Off_Resonance_Map_A11, 
                                        Off_Resonance_Map_B11, Off_Resonance_Map_A10, Off_Resonance_Map_A20, 
                                        Off_Resonance_Map_A21, Off_Resonance_Map_B21, Off_Resonance_Map_A22, 
                                        Off_Resonance_Map_B22, a, b, c, d, e, f, g, h, j, voxel_size):
        
        x_voxels = (np.arange(Off_Resonance_Map.shape[0]) * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[0] / 2)
        y_voxels = (np.arange(Off_Resonance_Map.shape[2]) * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2)
        z_voxels = (np.arange(Off_Resonance_Map.shape[1]) * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2)



        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, wspace=0.2)
        
        # Initialize the heatmaps with the middle slice
        heatmap1 = ax1.imshow(Off_Resonance_Map[:, :, int(Off_Resonance_Map.shape[2] / 2)],
                            cmap="viridis",extent=[x_voxels.min(), x_voxels.max(), z_voxels.min(), z_voxels.max()])
        plt.colorbar(heatmap1, ax=ax1)
        ax1.set_title("Off-Resonance 2-D Heat Map (slice)")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Z (mm)")

        vmin, vmax = heatmap1.get_clim()
        
        # Initialize the difference map
        diff_map = Off_Resonance_Map - (a*Off_Resonance_Map_A00+b*Off_Resonance_Map_A11
                                        +c*Off_Resonance_Map_B11+d*Off_Resonance_Map_A10
                                        +e*Off_Resonance_Map_A20+f*Off_Resonance_Map_A21
                                        +g*Off_Resonance_Map_B21+h*Off_Resonance_Map_A22
                                        +j*Off_Resonance_Map_B22)
        heatmap2 = ax2.imshow(diff_map[:, :, int(diff_map.shape[2] / 2)],
                            cmap="viridis", vmin=vmin, vmax=vmax,
                            extent=[x_voxels.min(), x_voxels.max(), z_voxels.min(), z_voxels.max()])
        plt.colorbar(heatmap2, ax=ax2)
        ax2.set_title("Shimmed Map (slice)")
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Z (mm)")
        
        # Histogram plot
        flat_non_corrected_roi = Off_Resonance_Map.flatten()
        flat_corrected_roi = diff_map
        flat_corrected_roi = flat_corrected_roi.flatten()
        ax3.hist(flat_non_corrected_roi, bins=50, alpha=0.5, label='Raw')
        ax3.hist(flat_corrected_roi, bins=50, alpha=0.5, label='Shimmed')
        ax3.set_title("Histogram of Frequencies (ROI)")
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Count")
        ax3.legend()

        def update(val):
            """Update function for the slice slider."""
            slice_index = int(slice_slider.val)
            heatmap1.set_data(Off_Resonance_Map[:, :, slice_index])
            diff_map_slice = diff_map[:, :, slice_index]
            heatmap2.set_data(diff_map_slice)
            
            # Remove all previous rectangles
            for patch in ax1.patches:
                patch.remove()
            for patch in ax2.patches:
                patch.remove()
            

        
        # Slider for selecting the slice
        ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
        slice_slider = Slider(ax_slice, 'y-Slice', 0, Off_Resonance_Map.shape[2] - 1, 
                            valinit=int(Off_Resonance_Map.shape[2] / 2), valstep=1)
        slice_slider.on_changed(update)
        
        plt.show()

    N = 8
    # Load data from each path
    volumes = []
    infos = []
    voxel_sizes = []
    for path in paths:
        volume, info, voxel_size , tr_time = load3d_dicom(path)
        volumes.append(volume)
        infos.append(info)
        voxel_sizes.append(voxel_size)
    
    volume_magnitude, volume_phase = volumes[0], volumes[1]
    voxel_size = voxel_sizes[0]

    volume_phase = phaze_transformation(volume_phase)
    complex_magnitude = volume_magnitude * np.exp(-1j * volume_phase)
    four_dimension_magnitude_data = np.array(complex_magnitude).reshape(
        (volume_magnitude.shape[0],volume_magnitude.shape[1],
         int(volume_magnitude.shape[2]/N),N), order="F")
    Off_Resonance_Map,zeroth_order_magnitude = FFT_Off_Resonance_Threshold(four_dimension_magnitude_data,tr_time)

    threshold = np.max(zeroth_order_magnitude)*0.1


    x_voxels = (np.arange(Off_Resonance_Map.shape[0]) * voxel_size[0] - voxel_size[0] * Off_Resonance_Map.shape[0] / 2)
    y_voxels = (np.arange(Off_Resonance_Map.shape[2]) * voxel_size[2] - voxel_size[2] * Off_Resonance_Map.shape[2] / 2)
    z_voxels = (np.arange(Off_Resonance_Map.shape[1]) * voxel_size[1] - voxel_size[1] * Off_Resonance_Map.shape[1] / 2)


    Off_Resonance_Map_A00 = shimming_base_function("A00",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A11 = shimming_base_function("A11",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B11 = shimming_base_function("B11",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A10 = shimming_base_function("A10",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A20 = shimming_base_function("A20",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A21 = shimming_base_function("A21",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B21 = shimming_base_function("B21",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_A22 = shimming_base_function("A22",x_voxels,y_voxels,z_voxels)
    Off_Resonance_Map_B22 = shimming_base_function("B22",x_voxels,y_voxels,z_voxels)


    A00_maps = Off_Resonance_Map_A00
    A11_maps = Off_Resonance_Map_A11
    B11_maps = Off_Resonance_Map_B11
    A10_maps = Off_Resonance_Map_A10
    A20_maps = Off_Resonance_Map_A20
    A21_maps = Off_Resonance_Map_A21
    B21_maps = Off_Resonance_Map_B21
    A22_maps = Off_Resonance_Map_A22
    B22_maps = Off_Resonance_Map_B22
    data = Off_Resonance_Map
    



    def ensure_same_shape(arrays):
        # Find the maximum shape
        max_shape = np.max([arr.shape for arr in arrays], axis=0)
        
        # Reshape arrays to the maximum shape by padding with NaNs
        padded_arrays = []
        for arr in arrays:
            padding = [(0, max_shape[i] - arr.shape[i]) for i in range(len(max_shape))]
            padded_array = np.pad(arr, padding, mode='constant', constant_values=np.nan)
            padded_arrays.append(padded_array)
            
        return padded_arrays

    # Ensure all arrays have the same shape
    data, A00_maps, A11_maps, B11_maps, A10_maps, A20_maps, A21_maps, B21_maps, A22_maps, B22_maps, zeroth_order_magnitude = ensure_same_shape(
        [data, A00_maps, A11_maps, B11_maps, A10_maps, A20_maps, A21_maps, B21_maps, A22_maps, B22_maps, zeroth_order_magnitude]
    )

    # Create mask
    mask = (~np.isnan(data) & ~np.isnan(A00_maps) & ~np.isnan(A11_maps) & 
            ~np.isnan(B11_maps) & ~np.isnan(A10_maps) & ~np.isnan(A20_maps) & 
            ~np.isnan(A21_maps) & ~np.isnan(B21_maps) & ~np.isnan(A22_maps) & 
            ~np.isnan(B22_maps) & (zeroth_order_magnitude > threshold))


    # Apply mask to all arrays
    matrix_3D_data = np.where(mask, data, np.nan)
    matrix_3D_A00_maps = np.where(mask, A00_maps, np.nan)
    matrix_3D_A11_maps = np.where(mask, A11_maps, np.nan)
    matrix_3D_B11_maps = np.where(mask, B11_maps, np.nan)
    matrix_3D_A10_maps = np.where(mask, A10_maps, np.nan)
    matrix_3D_A20_maps = np.where(mask, A20_maps, np.nan)
    matrix_3D_A21_maps = np.where(mask, A21_maps, np.nan)
    matrix_3D_B21_maps = np.where(mask, B21_maps, np.nan)
    matrix_3D_A22_maps = np.where(mask, A22_maps, np.nan)
    matrix_3D_B22_maps = np.where(mask, B22_maps, np.nan)

    # Apply mask to all arrays
    A00_maps = A00_maps[mask]
    A11_maps = A11_maps[mask]
    B11_maps = B11_maps[mask]
    A10_maps = A10_maps[mask]
    A20_maps = A20_maps[mask]
    A21_maps = A21_maps[mask]
    B21_maps = B21_maps[mask]
    A22_maps = A22_maps[mask]
    B22_maps = B22_maps[mask]
    data = data[mask]



    initial_guess = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    result = least_squares(residuals, initial_guess, loss='soft_l1', args=(A00_maps, A11_maps, B11_maps, A10_maps, A20_maps, A21_maps, B21_maps, A22_maps, B22_maps, data))

    """
                        --- Least Square Possible Loss Funtions ---

    linear (default) : rho(z) = z. Gives a standard least-squares problem.
    soft_l1 : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
    huber : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to soft_l1.
    cauchy : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.
    arctan : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to cauchy.
    """

    a, b, c, d, e, f, g, h, j = result.x

    #print(f"Optimal Shimming Values Change Relative to tune-up: A00 = {np.round(a,2)}, A11 = {-np.round(b*10**3/42.58,2)}, B11 = {np.round(c*10**3/42.58,2)}, A10 = {-np.round(d*10**3/42.58,2)}")
    #print(f"Optimal Shimming Values Change Relative to tune-up: A20 = {-np.round(e*10**6/42.58,2)} , A21 = {-np.round(f*10**6/42.58,2)} , B21 = {np.round(g*10**6/42.58,2)} , A22 = {-np.round(h*10**6/42.58,2)} , B22 = {np.round(j*10**6/42.58,2)}")

    if plotting == True :
        plot_interactive_heatmap_with_roi(matrix_3D_data, matrix_3D_A00_maps, matrix_3D_A11_maps, matrix_3D_B11_maps, matrix_3D_A10_maps, matrix_3D_A20_maps, matrix_3D_A21_maps, matrix_3D_B21_maps, matrix_3D_A22_maps, matrix_3D_B22_maps,
                                        a, b, c, d, e, f, g, h, j, voxel_size)
        
    # The parameters we want to actually retrieve 
    a = np.round(a)
    b = -np.round(b*10**3/42.58,2)
    c = np.round(c*10**3/42.58,2)
    d = -np.round(d*10**3/42.58,2)
    e = -np.round(e*10**6/42.58,2) 
    f = -np.round(f*10**6/42.58,2)
    g = np.round(g*10**6/42.58,2)
    h = -np.round(h*10**6/42.58,2)
    j = np.round(j*10**6/42.58,2)


    return(a,b,c,d,e,f,g,h,j)

def least_square_shim_masked_ROI(paths,plotting):


    return()



