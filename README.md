# Texture Analysis test tool for PET images

This tool is designed for only research purpose. Not for medical use. 
Due to the difference in metadata among manufactures, this tool only supports PET image dicom files which taken with a GE's PET device.
If your PET dicom files are taken by other than GE's, refer to the link below and modify \_\_get\_SUV\_conversion\_coeff function of VoiExtractionManager.py.
Or, simply return 1.0 at the function, if you do not need to convert intensity to SUV.

[Standardized_Uptake_Value](http://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_\(SUV\))


## Dependencies
* Python 3
* numpy
* sicpy 
* pandas 
* matplotlib 
* [pydicom1.0a+](https://github.com/pydicom/pydicom)

We recommend to install [Anaconda](https://www.continuum.io/downloads)(Python 3.x version) which is including all packages except pydicom.

### Installing pydicom from source
1. Download the pydicom source code from the ["Clone or download tab"]
2. Unzip the downloaded file
3. Execute the following command at unzipped  directory
```shell-session
$ python setup.py install
```
4. Execute following commands to check if the installation is succeeded
```shell-session
$ cd
$ python -c "import pydicom; print(pydicom.__version__)"
  1.0.0a1
```
The returned message will be 1.0.0a1 or higher if the installation is succeeded.

## Directory structure of data files
This tool assumes that data is saved in the following directory structure.
You can not use directory structures other than the following directory structure.

```
　data/
　　├PA0/    # Patient 0 data
　　│　└ST0/
　　│　　├SE0/　　# PET dicom image
　　│　　　　└0001.dcm, 0002.dcm ...
　　│　　├SE1/　　# CT dicom image
　　│　　　　└0001.dcm, 0002.dcm ...
　　│　　├SE2/　　# DICOM-RT of ROI1 (this filename must be "IM0")
　　│　　　　└IM0
　　│　　└SE3/　　# DICOM-RT of ROI2 (this filename must be "IM0")
　　│　　　　└IM0
　　└PA1/    # Patient 1 data
　　│　└ST0/
　　│　　├SE0/　　# PET dicom image
　　│　　　　└0001.dcm, 0002.dcm ...
　　│　　├SE1/　　# CT dicom image
　　│　　　　└0001.dcm, 0002.dcm ...
　　│　　├SE2/　　# DICOM-RT of ROI1 (this filename must be "IM0")
　　│　　　　└IM0
　　│　　└SE3/　　# DICOM-RT of ROI2 (this filename must be "IM0")
　　│　　　　└IM0
　　│　　　︙
　　│　　└SEn/　　# DICOM-RT of nth ROI (this filename must be "IM0")
　　│　　　　└IM0
　　︙
　　└PAn/    # nth Patient data
　　　└ST0/
　　　　　├SE0/　　# PET dicom image
　　　　　├SE1/　　# CT dicom image
　　　　　├SE2/　　# DICOM-RT of ROI1 (this filename must be "IM0")
　　　　　└SE3/　　# DICOM-RT of ROI2 (this filename must be "IM0")
```

## Usage
Based on the ROI information recorded in the DICOM-RT file, VOIs are extracted from PET dicom files by VoiExtractionManager.py. The extracted VOIs are saved under the ST0 directory. After that, texture values of each VOI are calculated by main.py.

### VOI extraction
VOI extraction is executed with the following command. Specify the path to the PA directory witht the "-d" option.
```shell-session
$ python VoiExtractionManager.py -d path/to/PA/directory/PA*
```
If you use "PA*" instead of "PA0" or "PA1" for the -d option, you can perform VOI extraction for multiple PA directories at once.

### Calculating texture values of each VOIs
By the following command the texture values of each VOI are calculated.
```shell-session
$ python main.py -d path/to/PA/directory/PA* -o ./results -n 64 --d_glcm 1 --d_ngtdm 1
```
The options are as follows.
* -d [src]　Specifies the path to target PA dirirectory
* -d [dst]　Specifies the path to output results
* -n [levels]　Specifies the number of rescaling levels. For example, the given value is 64, the intensity of each VOI is rescaled to range of 1-64. 
* --d_glcm [distance]　Specifies the distance paraameter of GLCM. istance>1 has not been impremented yet. 
* --d_ngtdm [distance] Specifies the distance paraameter of NGTDM. This parameter is variable, but if the given value is greater than 1, the calculation may fail because boundary value error handling is incomplete.
* --save_glcm_as_png　　Save GLCM as image data.


## Test with example data
An example dataset is included in the data directory.
The data is just to check if the installation of this program succeeded.
By executing following commands, an excel file is saved under the results directory if the installation is succeeded.
```shell-session
$ python VoiExtractionManager.py -d ./data/PA0
$ python main.py -d ./data/PA0
```

## Implemented texture features
* GLHA (Gray Level Histogram Analysis)			
    * kurtosis
    * mean
    * sd
    * skewness
    
* GLCM (Gray-Level Co-occurence Matrix)
    * contrast	
    * dissimilarity
    * entropy
    * homogeneity
    * inverse difference moment
    * maximum probability
    * uniformity
    
* GLCM (Gray-Level Co-occurence Matrix)
    * contrast	
    * dissimilarity
    * entropy
    * homogeneity
    * inverse difference moment
    * maximum probability
    * uniformity
    
* NGTDM
    * busyness	
    * coarseness	
    * complexity	
    * contrast	
    * strength
    
* NGTDM
    * high_intensity_emp
    * high_intensity_large_area_emp
    * high_intensity_small_area_emp
    * intensity_variability	
    * large_area_emp
    * low_intensity_emp
    * low_intensity_large_area_emp
    * low_intensity_small_area_emp
    * size_zone_variability	
    * small_area_emp
    * zone_percentage
    

## License
MIT License (see LICENSE file).


## References
Texture analysis on 18F-FDG PET/CT images to differentiate malignant and benign bone and soft-tissue lesions
DOI 10.1007/s12149-014-0895-9

Quantifying tumour heterogeneity in (18)F-FDG PET/CT imaging by texture analysis.
DOI 10.1007/s00259-012-2247-0

Textural analysis of pre-therapeutic \[18F\]-FET-PET and its correlation with tumor grade and patient survival in high-grade gliomas
DOI 10.1007/s00259-015-3140-4

Zone-size nonuniformity of 18F-FDG PET regional textural features predicts survival in patients with oropharyngeal cancer.
DOI 10.1007/s00259-014-2933-1

The promise and limits of PET texture analysis.
DOI 10.1007/s12149-013-0759-8

Non-Small Cell Lung Cancer Treated with Erlotinib: Heterogeneity of (18)F-FDG Uptake at PET-Association with Treatment Response and Prognosis.
DOI 10.1148/radiol.2015141309

Spatial-temporal \[¹⁸F\]FDG-PET features for predicting pathologic response of esophageal cancer to neoadjuvant chemoradiation therapy.
DOI 10.1016/j.ijrobp.2012.10.017

Three-dimensional positron emission tomography image texture analysis of esophageal squamous cell carcinoma: relationship between tumor 18F-fluorodeoxyglucose uptake heterogeneity, maximum standardized uptake value, and tumor stage.
DOI 10.1097/MNM.0b013e32835ae50c

Textural features of pretreatment 18F-FDG PET/CT images: prognostic significance in patients with advanced T-stage oropharyngeal squamous cell carcinoma.
DOI 10.2967/jnumed.112.119289
