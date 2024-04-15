# Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images

This is the official repository for our paper "Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images"

> **Detecting Abnormal Operations in Concentrated Solar Power Plants from Irregular Sequences of Thermal Images**
> Sukanya Patra, Nicolas Sournac, Souhaib Ben Taieb
>
> [[`Paper`](README.md)]

## Overview
Concentrated Solar Power (CSP) plants store energy by heating a storage medium with an array of mirrors that focus sunlight onto solar receivers atop a central tower. Operating at extreme temperatures exposes solar receivers to risks such as freezing, deformation, and corrosion. These problems can cause operational failures, leading to downtime or power generation interruptions, and potentially extensive equipment damage if not promptly identified, resulting in high costs. We study the problem of anomaly detection (AD) in sequences of thermal images collected over a span of one year from an operational CSP plant. These images are captured at irregular intervals ranging from one to five minutes throughout the day by infrared cameras mounted on solar receivers. Our goal is to develop an AD method to extract useful representations from high-dimensional thermal images, that is also robust to the temporal features of the data. This includes managing irregular intervals with temporal dependence between images, as well as accommodating non-stationarity due to a strong daily seasonal pattern. An additional challenge includes the coexistence of low-temperature anomalies resembling low-temperature normal images from the start and the end of the operational cycle alongside high-temperature anomalies. We first evaluate state-of-the-art deep anomaly detection methods for their performance in deriving meaningful image representations. Then, we introduce a forecasting-based AD method that predicts future thermal images from past sequences and timestamps via a deep sequence model. This method effectively captures specific temporal data features and distinguishes between difficult-to-detect temperature-based anomalies. Our experiments demonstrate the effectiveness of our approach compared to multiple SOTA baselines across multiple evaluation metrics. We have also successfully deployed our solution on six months of unseen data, providing critical insights to our industry partner for the maintenance of the CSP plant.

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
The simulated dataset can be downloaded using this [link](https://www.dropbox.com/scl/fo/u6bhb0xfvydtnvdbtmlc6/AN5boe8X1ZiUgSSRNKNZjTE?rlkey=amkjs2xcfmpp8odh7ssq02kir&dl=0). After downloading simulated_dataset.zip, extract the contents into the [`data`](data/) folder. The pickle files should thus be located at `<path-to-cloned-directory>\data\simulated_dataset\<name>.pickle`.

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