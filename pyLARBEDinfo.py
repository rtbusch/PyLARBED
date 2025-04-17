import numpy as np
from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import rotate
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from skimage.feature import peak_local_max
import pickle
from pyLARBED import ReadRaw, AlignByBeam, ApertureSumVariance, ApertureSum, getGridVectors_Adjust, get_gvectors, Precession, getGridVectors
import LucyRichardson as lr
from PeakTools import gaussian2D, fitPeak2DGauss

import matplotlib.pyplot as plt

class LARBEDAnalysis:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.average_raw = None
        self.average_aligned = None
        self.aligned = False
        self.g_grid = None
        self.g_vectors = None
        self.Larbed = []
        self.LarbedDeconvolved = []
        self.gg_pixels = []
        self.mtf_2d = None
        self.mixing = None
        self.center = (64, 64)
        self.IntRadius = 4
        self.IntZero = []

    def load_data(self,type=0):
        self.data = ReadRaw(self.file_name,type=type)
        self.center = (int(self.data.shape[2]//2), int(self.data.shape[3]//2))

    def align_data(self, iterations=3):
        aligned_matrix = np.copy(self.data)
        for _ in range(iterations):
            aligned_matrix = AlignByBeam(aligned_matrix, self.center[0], self.center[1], self.IntRadius)
        self.average_raw = np.mean(self.data, axis=(0, 1))
        self.data = aligned_matrix
        self.average_aligned = np.mean(self.data, axis=(0, 1))
        self.aligned = True

    def deconv_DPstack(self, mtf_2d, background=0, niter=100, varB=6.29, delta=0, A=1, g=580, m=None):
        deconvolved_stack = np.zeros_like(self.data)
        if m is None:
            m = np.sum(mtf_2d**2)/(mtf_2d.shape[0]*mtf_2d.shape[1])
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                print(i,j)
                deconvolved = lr.lucy_Richardson(self.data[i,j], mtf_2d, background, 
                                                    niter, varB, delta, A, g, m)
                deconvolved_stack[i,j] = deconvolved            
        self.data = deconvolved_stack
        return 
    
    def deconv_Larbedstack(self, background=0, niter=100, varB=6.29, delta=0.0, A=1, g=1, mtf_2d=None, pad_size=None, m=1):
        deconvolved_stack = np.zeros_like(self.Store_Larbed)
        deconvolved_Variance = np.zeros_like(self.Store_LarbedVariance)

        if mtf_2d is not None:
            self.mtf_2d = mtf_2d
        if pad_size is None:
            pad_size= 0
        #if self.mixing is None:
        self.mixing = m#np.sum(self.mtf_2d**2)/(self.mtf_2d.shape[0]*self.mtf_2d.shape[1])
        for i in range(len(self.Store_Larbed)):
            print(i)
            deconvolved  = []
            deconvolved  = np.pad(self.Store_Larbed[i], 
                        pad_width=((pad_size, pad_size), 
                                    (pad_size , pad_size)), 
                        mode='edge')
            deconvolved = lr.lucy_Richardson(deconvolved, self.mtf_2d, background, 
                                                niter, varB, delta, A, g, self.mixing)
            
            deconvolved_stack[i] = deconvolved[pad_size:self.Store_Larbed[i].shape[0]+pad_size,pad_size:self.Store_Larbed[i].shape[0]+pad_size]        
            deconvolved_Variance[i]=self.Store_LarbedVariance[i]+ self.mixing*g*deconvolved_stack[i] + varB
        self.LarbedDeconvolved = deconvolved_stack
        #self.Store_LarbedVariance = deconvolved_Variance
        return 


    def average_data(self):
        self.average_raw = np.mean(self.data, axis=(0, 1))
        self.average_aligned = self.average_raw
    
    def assign_gvector(self, g1, g2):
        self.gg_pixels = (g1, g2)

    def calculate_g_vectors(self, g1, g2):
        self.g_vectors = get_gvectors(g1, g2, self.grid)

    def calculate_grid_vectors(self, crop=1, order=3, Adjust=True):
        if Adjust:
            self.grid = getGridVectors_Adjust(self.center[0],self.center[1], 
                                              self.gg_pixels[0][0], self.gg_pixels[0][1], 
                                              self.gg_pixels[1][0], self.gg_pixels[1][1],  
                                              self.average_aligned, crop=crop, order=order)
        else:
            self.grid = getGridVectors(self.center[0],self.center[1], 
                                        self.gg_pixels[0][0], self.gg_pixels[0][1], 
                                        self.gg_pixels[1][0], self.gg_pixels[1][1],  
                                        self.average_aligned, order=order)

    def plot_averages(self, ratio=0.2):
        plt.figure()
        plt.imshow(self.average_raw, cmap='gray', vmin=0, vmax = np.max(self.average_aligned)*ratio)
        plt.title('Average Before Alignment')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(self.average_aligned, cmap='gray', vmin=0, vmax = np.max(self.average_aligned)*ratio)
        plt.title('Average After Alignment')
        plt.colorbar()
        plt.show()

    def ZeroLarbed(self,ratio=1):
        self.IntZero = ApertureSum(self.data, self.center[0], self.center[1], self.IntRadius)
        plt.figure()
        plt.imshow(self.IntZero, cmap='inferno', vmin=0, vmax = np.max(self.average_aligned)*ratio)
        plt.title('Larbed at' + str(self.center))
        plt.colorbar()
        plt.show()

    def plot_grid_vectors(self, ratio=0.2):
        plt.figure()
        plt.imshow(self.average_aligned, cmap='gray',vmax = np.max(self.average_aligned)*ratio)
        plt.title('Average After Alignment')
        plt.colorbar()

        for vector in self.grid:
            circle = plt.Circle((vector[1][1], vector[1][0]), radius=2, color='blue', fill=False)
            plt.gca().add_patch(circle)

        plt.show()


    def IntegrateLarbed(self, ri = 4, ro = (0,0), g=1,m=1,VarB=1,Variance=False):
        self.Store_Larbed = []
        self.Store_LarbedVariance = []
        removal = []
        for count, vector in enumerate(self.grid):
            try:
                New_Larbed = ApertureSum(self.data, vector[1][0], vector[1][1], ri, ro)
                self.Store_Larbed.append(New_Larbed)
                if Variance:
                    New_LarbedVariance = ApertureSumVariance(self.data, vector[1][0], vector[1][1], 
                                                             g, m, VarB, ri, ro) 
                    # 1, 0.1319299, 6.2968867, 4, (7, 9)
                    self.Store_LarbedVariance.append(New_LarbedVariance)
            except:
                removal.append(count)
                continue
        for i in sorted(removal, reverse=True):
            self.g_vectors.pop(i)
            self.grid.pop(i)

    def plot_IntegratedLarbed(self, index, Variance = False):
        if Variance:
            plt.figure()
            plt.imshow(self.Store_LarbedVariance[index], cmap='gray')
            plt.title(f"Integrated Larbed Variance at g-vector: {self.g_vectors[index]}")
            plt.colorbar()
            plt.show()
        if index == 'all':
            for i in range(len(self.Store_Larbed)):
                plt.figure()
                plt.imshow(self.Store_Larbed[i], cmap='gray')
                plt.title(f"Integrated Larbed at g-vector: {self.g_vectors[i]}")
                plt.colorbar()
                plt.show()
        else:
            plt.figure()
            plt.imshow(self.Store_Larbed[index], cmap='gray')
            plt.title(f"Integrated Larbed at g-vector: {self.g_vectors[index]}")
            plt.colorbar()
            plt.show()


    def deconvolve_image(self, index, niter=100, varB=6.29, delta=0, A=1, g=580):
        deconv = lr.lucy_Richardson(self.Store_Larbed[index], self.mtf_2d, background=np.min(self.Store_Larbed[0]), niter=niter, varB=varB, delta=delta, A=A, g=g, m=self.mixing)
        return deconv
    
    def plot_larbed(self, rotation = -22, order_option=1, order=6, Circle_radius=8, a_expansion=1.6, ratio = 0.2, Variance=False):
        fig, ax = plt.subplots(figsize=(40, 40))

        for count, vector in enumerate(self.grid):
            vector2 = self.grid[count]
            if order_option == 0:
                if self.g_vectors[count][0] < -order or self.g_vectors[count][1] < -order or self.g_vectors[count][2] < -order or self.g_vectors[count][0] > order or self.g_vectors[count][1]  > order or g_vectors[count][2] >order: 
                    continue
            if order_option == 1:
                if np.sqrt(self.g_vectors[count][0]**2 + self.g_vectors[count][1]**2 + self.g_vectors[count][2]**2) > order: 
                    continue
            hex = Circle((vector2[1][0]*a_expansion, vector2[1][1]*a_expansion), radius=Circle_radius, fill=False, 
                         alpha=1, edgecolor='w', linewidth=2)
            ax.add_patch(hex)

            try:
                if Variance:
                    im = ax.imshow(rotate(np.flipud(self.Store_LarbedVariance[count]), rotation, reshape=False), extent=[vector2[1][0]*a_expansion-Circle_radius, 
                                                                                            vector2[1][0]*a_expansion+Circle_radius, 
                                                                                            vector2[1][1]*a_expansion-Circle_radius, 
                                                                                            vector2[1][1]*a_expansion+Circle_radius], 
                                                                                            cmap='inferno', alpha=1, vmin=0, 
                                                                                            vmax=max(self.Store_LarbedVariance[count].flatten(), -20)[-20]*ratio)
                    im.set_clip_path(hex)
                else:
                    im = ax.imshow(rotate(np.flipud(self.Store_Larbed[count]), rotation, reshape=False), extent=[vector2[1][0]*a_expansion-Circle_radius, 
                                                                                            vector2[1][0]*a_expansion+Circle_radius, 
                                                                                            vector2[1][1]*a_expansion-Circle_radius, 
                                                                                            vector2[1][1]*a_expansion+Circle_radius], 
                                                                                            cmap='inferno', alpha=1, vmin=0, 
                                                                                            vmax=np.partition(self.Store_Larbed[count].flatten(), -20)[-20]*ratio)
                    im.set_clip_path(hex)
            except:
                continue

        ax.autoscale_view()
        ax.set_aspect('equal')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        fig.canvas.draw()
        plt.show()
    
    def save_larbed(self, filename):
        np.save(filename + '_Store_Larbed.npy', self.Store_Larbed)
        np.save(filename + '_Store_LarbedDeconvolved.npy', self.LarbedDeconvolved)
        np.save(filename + '_Store_LarbedVariance.npy', self.Store_LarbedVariance)
        np.save(filename + '_g_vectors.npy', self.g_vectors)

class LARBEDCalibration:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.crop_data = []
        self.probe_image = None
        self.mtf_2d = None
        self.peaks = None
        self.roi_size = None

    def load_data(self, type=0):
        self.data = ReadRaw(self.file_name,type=type)
        self.center = (int(self.data.shape[2]//2), int(self.data.shape[3]//2))
        plt.imshow(self.data[int(self.data.shape[0]//2),int(self.data.shape[1]//2)], cmap='gray')

    def meanimage(self):
        return np.mean(self.data, axis=(0, 1))
        
    def find_peaks(self, start=1,end=255):
        x_locations = []
        y_locations = []
        self.peaks = []
        self.roi_size = (start,end)


        for i in range(start, end):
                print(i)
                for j in range(start, end):
                        patch = self.data[i, j, :, :]
                        params2 = fitPeak2DGauss(patch, gaussian2D)
                        _, x, y, _ = params2
                        x_locations.append(x)
                        y_locations.append(y)
        
        self.peaks = np.array([x_locations, y_locations])
        return 

    def crop_peaks(self, crop_size=20):
        temp_size = self.roi_size[1] - self.roi_size[0]
        self.crop_data = np.zeros((temp_size, temp_size, crop_size, crop_size))
        for i in range(temp_size):
            for j in range(temp_size):
                x_center = self.peaks[0][i * temp_size + j]
                y_center = self.peaks[1][i * temp_size + j]
                
                # Define the cropping bounds
                x_start = int(np.floor(x_center - crop_size / 2))
                x_end = x_start + crop_size
                y_start = int(np.floor(y_center - crop_size / 2))
                y_end = y_start + crop_size
                
                # Extract the patch and apply sub-pixel alignment
                patch = self.data[i + self.roi_size[0], j + self.roi_size[0], y_start:y_end, x_start:x_end]
                aligned_patch = zoom(patch, (1, 1), order=3, mode='nearest', prefilter=True)
                
                # Ensure the cropped patch is of the correct size
                if aligned_patch.shape == (crop_size, crop_size):
                    self.crop_data[i, j] = aligned_patch
        return

    def calculate_average_step_length(self):
        self.average_step_length = None
        x_grid = np.zeros((self.roi_size[1] - self.roi_size[0], self.roi_size[1] - self.roi_size[0]))
        y_grid = np.zeros((self.roi_size[1] - self.roi_size[0], self.roi_size[1] - self.roi_size[0]))
        # Reshape the x and y locations into a grid
        x_grid = self.peaks[0].reshape(self.roi_size[1] - self.roi_size[0], self.roi_size[1] - self.roi_size[0])
        y_grid = self.peaks[1].reshape(self.roi_size[1] - self.roi_size[0], self.roi_size[1] - self.roi_size[0])

        # Calculate differences along the horizontal (axis=1) and vertical (axis=0) directions
        dx_horizontal = np.diff(x_grid, axis=1)
        dy_horizontal = np.diff(y_grid, axis=1)
        dx_vertical = np.diff(x_grid, axis=0)
        dy_vertical = np.diff(y_grid, axis=0)

        # Calculate Euclidean distances for horizontal and vertical steps
        xtilt_mag = np.sqrt(dx_horizontal**2 + dy_horizontal**2)
        ytilt_mag = np.sqrt(dx_vertical**2 + dy_vertical**2)

        # Combine all step lengths
        all_step_lengths = np.concatenate([xtilt_mag.flatten(), ytilt_mag.flatten()])

        # Calculate the average step length
        self.average_step_length = np.mean(all_step_lengths)
        print(f"Average step length: {self.average_step_length}")
        return

    def interpolate(self):
        self.probe_image = zoom(np.mean(self.crop_data,axis=(0,1)), 
                                (1/self.average_step_length ,
                                           1/self.average_step_length ), order=3)
        
        #self.probe_image = zoom(np.mean(self.crop_data,axis=(0,1)), 
        #                        (1/np.sqrt(2*self.average_step_length*self.average_step_length) ,
        #                                   1/np.sqrt(2*self.average_step_length*self.average_step_length) ), order=3)
        return

    def calculate_mtf(self, pad_size = 256):
        self.mtf_2d  = np.pad(self.probe_image, 
                              pad_width=((0, pad_size - self.probe_image.shape[0]), 
                                         (0, pad_size - self.probe_image.shape[1])), 
                              mode='constant', 
                              constant_values=0)

        self.mtf_2d = np.abs(np.fft.fft2(self.mtf_2d))
        self.mtf_2d = (self.mtf_2d-np.min(self.mtf_2d))/(np.max(self.mtf_2d)-np.min(self.mtf_2d))
        return
