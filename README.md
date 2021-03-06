# Reconstructing the house from the ad: Structured prediction on real estate classifieds

This repository contains the code used for dependency parsing and information about how to obtain the dataset presented in the work:

[Reconstructing the house from the ad: Structured prediction on real estate classifieds](https://bekou.github.io/papers/eacl2017/bekoulis-eacl2017.pdf)

The dataset includes 2,318 manually annotated property advertisements from a real estate company.

If you use part of the code or the dataset please cite:

```  
@InProceedings{E17-2044,
  author = 	"Bekoulis, Giannis
		and Deleu, Johannes
		and Demeester, Thomas
		and Develder, Chris",
  title = 	"Reconstructing the house from the ad: Structured prediction on real estate classifieds",
  booktitle = 	"Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"274--279",
  location = 	"Valencia, Spain",
  url = 	"http://aclweb.org/anthology/E17-2044"
}
```
and 
```
@article{BEKOULIS2018100,
title = "An attentive neural architecture for joint segmentation and parsing and its application to real estate ads",
journal = "Expert Systems with Applications",
volume = "102",
pages = "100 - 112",
year = "2018",
issn = "0957-4174",
doi = "https://doi.org/10.1016/j.eswa.2018.02.031",
url = "http://www.sciencedirect.com/science/article/pii/S0957417418301192",
author = "Giannis Bekoulis and Johannes Deleu and Thomas Demeester and Chris Develder"
}
```


### Pre-requisites ###

The code is written for Python 2.7. Some of the python packages needed to run these files, best installed using *pip*.

* scikit-learn (machine learning)
* pandas (Data manipulation)
* pandas_confusion (performance measures)

#### Dependency parser ####

In the repository, one can find the 4 models (Threshold, Edmond, Structured Prediction via the Matrix-Tree Theorem (MTT), Transition-based) that we have developed for dependency parsing. One should run the *run_script.py* file that serves as a main function.

#### Dataset ####

To obtain the anonymized dataset fill in and sign [this](https://github.com/bekou/ad_data/raw/master/agreement/data-agreement.pdf) form. Send it also via email to giannis.bekoulis@gmail.com. Follow the instructions and we will get back to you as soon as possible with information about how to download the anonymized dataset. 
