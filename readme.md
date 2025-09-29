# Code for  OSA Detection Based on SKTS-OPS-SPCL Method
This repository provides the code for paper:
An Ordinal Pattern Similarity-Guided Supervised Prototype Contrastive Learning Framework with Enhanced Token Selection Module for Sleep Apnea Detection Based on Wearable PPG Bracelet <br>
<img alt="Fig.1. The overall structural schematic diagram of the proposed OPS-SKTS-SPCL method" src="C:\Users\Administrator\Desktop\github\results\Fig1.bmp" title="the proposed OPS-SKTS-SPCL method"/>

## Introduction
###  we propose an ordinal pattern similarity-guided supervised prototype contrastive learning (ops-spcl) framework with enhanced token selection module. 
On the wearable bracelet PPG dataset, the proposed model performs well in per-segment detection, achieving 72.4% sensitivity and a 0.687 F1-score. 
By analyzing SA detection rates across different population groups, we demonstrate that it effectively focuses on the hard samples of mild to moderate patients. 
Moreover, these findings highlight the generality of its strong generalization capabilities across multiple datasets. 
These results suggest that our approach holds significant potential for large-scale SA detection.

## Dataset
###  **Wearable Bracelet database** <br> 
It contained PPG signals from 92 cases of wearable bracelets, with synchronized PSG serving as the gold standard for annotating the subjects' sleep states. The sampling frequency of the wearable bracelets was 100 Hz.<br>
### **Apnea-PPG database** <br> 
It includes 110 PPG records in a single-lead at night. It was collected from The Sixth Affiliated Hospital of Sun Yat-sen University.The sampling frequency of the wearable bracelets was 100 Hz.<br>
### **Apnea-ECG database** <br> 
The data consist of 70 records, divided into a learning set of 35 records (a01 through a20, b01 through b05, and c01 through c10), and a test set of 35 records (x01 through x35).The sampling frequency of the wearable bracelets was 100 Hz.<br>

## Input Signal
The CSV file contains six columns: <br> 
id – the identifier.<br>
label_SA – values are either 0 or 1, with a shape of (1,).<br>
RRdata – with a shape of (500, 1).<br>
entropy – with a shape of (24, 1).<br>
entropy_sum – with a shape of (1, 1).<br>
sort – with a shape of (24, 1).<br>

## Validity Analysis of Model
By leveraging the OPS method, it combines the explicit construction of hard negative sample pairs and easy positive sample pairs with the supervised prototype contrastive objective to improve intra-class compactness and inter-class separability.<br>
<img alt="Fig.2. Feature visualization depicting representation learning process." src="C:\Users\Administrator\Desktop\github\results\Fig2.bmp" title="Feature visualization"/>

## Requirements
This project requires Python 3.8+ and the dependencies listed in requirements.txt<br>
You can install all dependencies with:<br>
pip install -r requirements.txt

## Usage
### Then, follow these steps and you will get the OSA detection model.

1. Data Preprocessing. Please refer to Section III. METHOD  A. The Dataset and Preprocessing of the paper for data preprocessing.
2. Run main.py. This python file train SKTS-OPS-SPCL network for per-segment OSA detection and per-recording OSA detection.
Noted that the optimization function of Keras and TensorFlow is slightly different in different versions. Therefore, to reproduce our results, it suggests that using the same version of Keras and TensorFlow as us, in our work the version of Keras is 2.4.3 and TensorFlow is 2.4.1.

## Email:
If you have any questions, please email to: liugzh3@mail.sysu.edu.cn

## Citation
If you find the SKTS-OPS-SPCL method useful in your research, we would appreciate it if you cite it.<br>
@article{,<br>
  title={An Ordinal Pattern Similarity-Guided Supervised Prototype Contrastive Learning Framework with Enhanced Token Selection Module for Sleep Apnea Detection Based on Wearable PPG Bracelet},<br>
  author={Weiyan Qiu,  Zhuo Chen, Yanxun Lu, Changhong Wang and Guanzheng Liu},<br>
  journal={},<br>
  year={2025},<br>
}

## Licence
For academtic and non-commercial usage only
