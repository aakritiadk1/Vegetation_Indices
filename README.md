# Vegetation Index Comparison Across Landsat 8 and Landsat 9

This repository compares vegetation indexes derived from Landsat 8 and Landsat 9 optical satellite imagery. It includes definitions, formulations, and implementations of these indexes for each satellite platform.

## Project Structure

### codes 

Contains `data_download.ipynb`:
- Jupyter Notebook for downloading satellite data.
- Usage: Provide a shapefile from the `shapefile` folder to define the study area.

### shapefile 

Contains the study area shapefile used in `data_download.ipynb`.

### functions.py

Contains reusable functions used across the project.

### metricscalcandplot.ipynb

Includes:
- Comparison and plotting of NDVI, EVI, and NDWI indexes across Landsat 8, Landsat 9, and Sentinel 2.
- Detailed analysis and visualization of index variations.

## Vegetation Index 

### NDVI (Normalized Difference Vegetation Index)

NDVI is a measure of vegetation greenness or density. It is calculated using the formula:

NDVI = (NIR - Red) / (NIR + Red)

Where NIR (Near-Infrared) and Red are reflectance values from corresponding bands.

### EVI (Enhanced Vegetation Index)
EVI enhances the sensitivity of vegetation index calculations to canopy structural variations and atmospheric conditions. It is calculated as:
EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))

### NDWI (Normalized Difference Water Index)
NDWI is used to highlight water bodies and is calculated using:
NDWI = (Green - NIR) / (Green + NIR)
