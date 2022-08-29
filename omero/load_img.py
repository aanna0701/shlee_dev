from PIL import Image
from typing import List
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

class Omero_Image:
    '''
    Omero Image
    Load raw image and visualize
    
    Author: Seunghoon Lee
    Date: 2022.08.29
    Ver: 1.0.0
    '''
    
    COLORS = ['gray', 'Blues_r', 'Greens_r']
    
    def __init__(self, path: str, tag: str = None, cut_off: float = 1e-5) -> None:
        '''
        [args]
        path: Path of raw image
        tag: Tag of instance
        cut_off: cut off parameter for auto-thresholding
        
        [variables]
        _img: NumPy array object of the image (C, Z, H, W)
        _shape: Shape of _img
        _hist_list: Histogram list for each channel and each z-slice
        _min_th_list, _max_th_list: Min threshold and max threshold for each channel
        '''
        
        self._tag = tag
        self._path = path
        img = self.tif_to_np(self._path) # (C*Z, H, W)
        self._img = self.reshape_3d_to_4d(img) # (C, Z, H, W) 
        self._shape = self._img.shape
        self._cut_off = cut_off

        self._hist_list = list()
        for i in range(self._shape[0]):
            tmp = list()
            for j in range(self._shape[1]):
                tmp.append(self.hist(i, j))
            self._hist_list.append(tmp)
            
        self._min_th_list, self._max_th_list = self.find_min_max_along_whole_imgs(self._hist_list, cut_off=self._cut_off, type='less')
    
    
    def tif_to_np(self, path: str) -> np.ndarray:
        '''
        Convert .tif image to NumPy array
        
        [args]
        path: path of .tif image
        
        [return]
        Images in the form of NumPy array
        '''
        
        tif_img = Image.open(path)
        np_img = list()
        for i in range(tif_img.n_frames):
            tif_img.seek(i) # (H, W)
            np_img.append(np.array(tif_img))
        return np.array(np_img) # (C*Z, H, W)
    
    
    def reshape_3d_to_4d(self, img: np.ndarray, num_channels: int = 3) -> np.ndarray:
        '''
        Reshaep 3-D NumPy array to 4-D
        
        [args]
        img: input NumPy array (C*Z, H, W)
        num_channels: the number of channels of input
        
        [return]
        Reshaped NumPy array (C, Z, H, W)
        '''
        
        CZ, H, W = img.shape
        out = img.reshape((num_channels, CZ//num_channels, H, W)) # (C, Z, H, W)
        return out
    
    
    def hist(self, channel: int = 0, z:int = 0) -> list:
        '''
        Calculate histogram of pixel values
        
        [args]
        channel: channel of image for which to obtain histogram
        z: z-slice to get histogram
        density: whether obtaining a normalized histogram
        auto_th: whether auto-thresholding
        cut_off: cut off value of auto-thresholding
        plot: plotting histogram
        
        [return]
        norm_hist: normalized histogram
        min_value: min value of input
        max_value: max value of input
        '''
        
        img = self._img[channel, z]
        max_value = np.max(img)
        min_value = np.min(img)
        num_bins= int(max_value - min_value + 1) # determine range (x-axis)
    
        hist = cv2.calcHist([img], [0], None, [num_bins], [int(min_value), int(max_value)+1]) 
        resolution = self._shape[-1] * self._shape[-2]
        norm_hist = hist / resolution
        
        return norm_hist, min_value, max_value
    
    
    def find_min_max_along_whole_imgs(self, hist_list: List[List[np.ndarray]], type: str = 'more', cut_off=1e-4) -> list:
        '''
        Determine the min and max threshold for all images.
        A representative threshold is determined through thresholds of all z-slice for each channel
        
        [args]
        hist_list: Histogram list for each channel and each z-slice
        type: How to determine the threshold at each z-slice
        cut_off: cut off parameter for auto-thresholding
        
        [return]
        min and max thresholds for each channel
        '''
        
        assert type in ['more', 'mean', 'less'], 'Unvalid Type !!!'
        
        if type == 'more':
            min_th_list = list()
            max_th_list = list()
            for hist_per_channel in hist_list:
                min_th = 9e5
                max_th = 0
                
                for hist_per_z, min_value, max_value in hist_per_channel:    
                    th = self.auto_thershold(hist_per_z, min_value, max_value, cut_off)
                    if th > max_th: max_th = th
                    if min_value < min_th: min_th = min_value
                
                min_th_list.append(min_th)
                max_th_list.append(max_th)
        
        elif type == 'less':
            min_th_list = list()
            max_th_list = list()
            for hist_per_channel in hist_list:
                min_th = 0
                max_th = 9e5
                
                for hist_per_z, min_value, max_value in hist_per_channel:    
                    th = self.auto_thershold(hist_per_z, min_value, max_value, cut_off)
                    if th < max_th: max_th = th
                    if min_value > min_th: min_th = min_value
                
                min_th_list.append(min_th)
                max_th_list.append(max_th)
                
        return min_th_list, max_th_list 
    
    
    def auto_thershold(self, hist: np.ndarray, min_value: float, max_value: float, cut_off: float = 1e-5) -> None:
        '''
        Threshold is automatically determined using cut off parameters.
        Navigate in descending order
        
        [args]
        hist: input NumPy list
        min_value: minimum value
        max_value: maximum value
        '''
        
        for i in range(max_value - min_value + 1):
            if hist[-(i+1)] >= cut_off:
                return int(max_value-i-min_value)
    
    
    def img_show(self, channel: int = 0, z: int = 0, normalize: bool = True) -> None:
        '''
        Visualize the image
        
        [args]
        channel: channels to visualize
        z: z-slice to visualize
        normalize: whether to normalize (auto-normalizing)
        '''
        
        if not normalize:
            img = self._img[channel, z]
        else:
            img = self.thresholding_img(self._img[channel, z], channel=channel)
        
        plt.imshow(img, cmap=Omero_Image.COLORS[channel])
        plt.axis('off')
        plt.margins(0, 0)
    
    
    def thresholding_img(self, img: np.ndarray, channel: int ) -> np.ndarray:
        '''
        Thresholding for each channel
        
        [args]
        img: input image
        channel: channel to thresholding
        
        [return]
        Threshold processed image
        '''
        
        img_c = copy.deepcopy(img)
        img_c[img_c >= self._max_th_list[channel]] = self._max_th_list[channel]
        img_c[img_c <= self._min_th_list[channel]] = self._min_th_list[channel]
        
        return img_c
    
    
    def __repr__(self) -> str:
        return f'tag: {self._tag}\npath: {self._path}\nshape: {self._shape}\ncut off: {self._cut_off}'
        
        
    def __str__(self) -> str:
        return f'tag: {self._tag}\npath: {self._path}\nshape: {self._shape}\ncut off: {self._cut_off}'


def omero_10x(path: str) -> Omero_Image:
    return Omero_Image(path=path, tag='10x', cut_off=1e-5)


def omero_20x(path: str) -> Omero_Image:
    return Omero_Image(path=path, tag='20x', cut_off=1e-8)