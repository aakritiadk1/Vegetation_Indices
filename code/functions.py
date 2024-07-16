#%%
import geemap
from geopandas import gpd   
import ee
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.plot import show
from pathlib import Path
from skimage.transform import resize
#%%
#to save the data collection
def save_data(out_dir, image, filename, ROI):
    file_dir = out_dir/filename
    basename = filename.split('.')[0].split('_')[0]
    if basename =='landsat':
        geemap.download_ee_image(image, filename = file_dir, scale=30, region=ROI, crs='EPSG:4326')
        
    elif basename == 'sentinel':
        geemap.download_ee_image(image, filename = file_dir, scale=10, region=ROI, crs='EPSG:4326')
    
# %%
def get_eesupported_roi(shp_file):
    shp = gpd.read_file(shp_file)
    shp = shp.to_crs(epsg=4326)
    roi_geom = shp.geometry.values[0]
    roi_geojson = roi_geom.__geo_interface__
    roi_ee = ee.Geometry(roi_geojson)
    return roi_ee

#%%
def read_raster(raster_path, data_dir):
    with rasterio.open(raster_path) as src:
        profile = src.profile
        basename = os.path.basename(raster_path).split('.')[0]
        basename_split = basename.split('_')[0]
        
        if basename_split == 'landsat':
            red = src.read(4).astype(np.float32)
            nir = src.read(5).astype(np.float32)
            blue = src.read(2).astype(np.float32)
            green = src.read(3).astype(np.float32)

            # Replace NaN values with mean of each band
            red = np.where(np.isnan(red), np.nanmean(red), red)
            nir = np.where(np.isnan(nir), np.nanmean(nir), nir)
            blue = np.where(np.isnan(blue), np.nanmean(blue), blue)
            green = np.where(np.isnan(green), np.nanmean(green), green)
            
            
            print("Calculation of NDVI, EVI, NDWI in progress... of the raster file: {basename} ")
            #NDVI Calculation
            ndvi = (nir - red) / (nir + red)
            
            #Calculate EVI
            evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
            
            #Calculate NDWI
            ndwi = (green - nir) / (green + nir)
            
            # Create NDVI, EVI, NDWI folders inside the data directory if they don't exist
            ndvi_dir = data_dir / 'NDVI'
            evi_dir = data_dir / 'EVI'
            ndwi_dir = data_dir / 'NDWI'
            
            os.makedirs(ndvi_dir, exist_ok=True)
            os.makedirs(evi_dir, exist_ok=True)
            os.makedirs(ndwi_dir, exist_ok=True)
            
            # Save NDVI, EVI, NDWI raster files
            ndvi_path = ndvi_dir / f'{basename}NDVI.tif'
            evi_path = evi_dir / f'{basename}EVI.tif'
            ndwi_path = ndwi_dir / f'{basename}NDWI.tif'

            
            #Call the save_raster function to save the raster files
            save_raster(ndvi_path, ndvi, src)
            save_raster(evi_path, evi, src)
            save_raster(ndwi_path, ndwi, src)        
            
def save_raster(output_path, data, src):
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
        print(f"Raster file saved at: {output_path}")
# %%
# Enable LaTeX font rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_rasters_with_custom_titles(data_dir, custom_titles, colorbar_label):
    # Get all .tif files in the directory
    tif_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.tif')]

    # Plotting the rasters
    num_images = len(tif_files)
    cols = 2  
    rows = (num_images + 1) // cols  

    # Define the target size for all images
    target_size = (512, 512) 

    fig, axs = plt.subplots(rows, cols, figsize=(14, 7 * rows))

    for i, tif_file in enumerate(tif_files):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]

        dataset = rasterio.open(tif_file)
        raster_data = dataset.read(1)
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        # Resize raster data to target size
        raster_resized = resize(raster_data, target_size, mode='reflect', anti_aliasing=True)
        
        img = ax.imshow(raster_resized, cmap='terrain', extent=extent)
        ax.set_title(rf'\textbf{{{custom_titles[i]}}}', fontsize=14)
        ax.set_xlabel(r'\textbf{Longitude}', fontsize=10)
        ax.set_ylabel(r'\textbf{Latitude}', fontsize=10)
        ax.yaxis.set_tick_params(rotation=90)
        
        # Add colorbar to each image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colorbar = plt.colorbar(img, cax=cax, orientation='vertical')
        colorbar.set_label(colorbar_label)
        
        dataset.close()

    # Adjust layout
    plt.tight_layout()
    plt.show()