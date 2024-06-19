# Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images

This is the official repository for our paper "Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images"

> **Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images**
> Sukanya Patra, Nicolas Sournac, Souhaib Ben Taieb
> In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24)  
> [[`Paper`](README.md)]

## Overview
Concentrated Solar Power (CSP) plants store energy by heating a storage medium with an array of mirrors that focus sunlight onto solar receivers atop a central tower. Operating at high temperatures these receivers face risks such as freezing, deformation, and corrosion, leading to operational failures, downtime, or costly equipment damage. We study the problem of anomaly detection (AD) in sequences of thermal images collected over a year from an operational CSP plant. These images are captured at irregular intervals ranging from one to five minutes throughout the day by infrared cameras mounted on solar receivers. Our goal is to develop a method to extract useful representations from high-dimensional thermal images for AD. It should be able to handle temporal features of the data, which include irregularity, temporal dependency between images and non-stationarity due to a strong daily seasonal pattern. The co-occurrence of low-temperature anomalies that resemble normal images from the start and the end of the operational cycle with high-temperature anomalies poses an additional challenge. We first evaluate state-of-the-art deep image-based AD methods, which have been shown to be effective in deriving meaningful image representations for the detection of anomalies. Then, we introduce a forecasting-based AD method that predicts future thermal images from past sequences and timestamps via a deep sequence model. This method effectively captures specific temporal data features and distinguishes between difficult-to-detect temperature-based anomalies. Our experiments demonstrate the effectiveness of our approach compared to multiple SOTA baselines across multiple evaluation metrics. We have also successfully deployed our solution on five months of unseen data, providing critical insights to our industry partner for the maintenance of the CSP plant.

## Architecture

<p align="center">
<img src="assests/ForecastAD.jpg" alt="ForecastAD" style="width:700px;"/>
</p>

Illustration of the end-to-end architecture of ForecastAD. The model is trained to forecast the next image in the sequence given a context embedding $c_i$ of $K$ prior samples obtained using a sequence-to-sequence model. For a sample $(x_i, t_i, y_i) \in \mathcal{D}$, we concatenate the image embedding with the sum of inter-arrival time $\tau_i$ and interval since the start of operation $\delta_i$. The anomaly score is assigned as the $\ell_2$ distance between the forecasted and original images.


## Installation
This code is written in `Python 3.9` and requires the packages listed in [`environment.yml`](environment.yml).

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-cloned-directory>
conda env create --file environment.yml
conda activate csp_env
```

## Dataset
The simulated dataset can be downloaded using this [link](https://www.dropbox.com/scl/fi/w8sa7i6lfimq85hfhywsw/simulated_dataset.zip?rlkey=m037crc0oto7c3iceqog5wlcc&st=bqa9fh91&dl=0). After downloading simulated_dataset.zip, extract the contents into the [`data`](data/) folder. The pickle files should thus be located at `<path-to-cloned-directory>\data\simulated_dataset\<name>.pickle`.

We further provide the train, test and validation split used in our experiments in the [`data`](data/) folder.

## Running experiments

To run an experiment create a new configuration file in the [`configs`](configs/) directory. The experiments can be can run using the following command:

```
cd <path-to-cloned-directory>\src
python  main.py --exp_config ..\configs\<config-file-name>.json
```

We provide the configuration files for the running ForecastAD with sequence length 30 and image size 256

## License

This project is under the MIT license.
