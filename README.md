# Lung cancer detection using a 3D CNN 

*Authors: Rick Beeloo, Thomas Reinders and Koen van der Heide*
## Introduction ###

Lung cancer is the leading cause of cancer-related deaths for both men and women across the developed world. Despite tremendous efforts to treat this cancer, the overall 5-year survival for all stages is dismally low at 15% [[1](https://www.ncbi.nlm.nih.gov/pubmed/9187198)]. In the daily practice of radiology, medical images from different modalities are read and interpreted by radiologists. Usually radiologists must analyze and evaluate these images comprehensively in a short time. But with the advances in modern medical technologies, the amount of imaging data is rapidly increasing. For example, CT examinations are being performed with thinner slices than in the past. The reading and interpretation time of radiologists will mount as the number of CT slices grows. Machine learning provides an effective way to automate the analysis and diagnosis for medical images. It can potentially reduce the burden on radiologists in the practice of radiology[[2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3372692/ "2")]. Although automatic cancer detection is faster than human curation, sequential machine learning is still time consuming and computer costly.  We therefore utilize graphics processing units (GPUs) to speed up this time consuming process and quantitively asses its profit. 

## Getting started ##
You can clone this repository using ```git clone https://github.com/rickbeeloo/cnn```. This code is written in ```python 3.6``` and uses several dependencies:

 - [Tensorflow V1.7 GPU](https://www.tensorflow.org/) (+dependencies see website)
 - [Pandas](https://pypi.python.org/pypi/pandas/0.17.1/)
 - [OpenCV-python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
 - [Dicom](https://pypi.python.org/pypi/dicom)
 
 All can be easily installed using `pip install`

## Data ##
For this project we used the [Sience Bowl](https://en.wikipedia.org/wiki/National_Science_Bowl "sience bowl") lung cancer data, which is available [here](https://www.kaggle.com/c/data-science-bowl-2017/data "here"): 
- `stage1.7z` - contains all CT images for the first stage of the competition
- `sample_images.7z` - a smaller subset set of the full dataset, provided for people who wish to preview the images before downloading the large file.
- `stage1_labels.csv` - contains the cancer ground truth for the stage 1 training set images

***Note:** We use the sample_images.7z only for illustration purposes in the code below as the small size makes it easily repeatable*

## Code ##
**Source and adjustments**<br/>
We adjusted [this code](https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet/notebook "this code") from Sentdex and therefore credits to him.  
The adjustments made:
- Added the ability to test the model on new data
- Added thorough explanation for each of the parameters required
- Added the picture ids to the test output to be able to trace the predictions back to the corresponding images
- Wrote the code Object Orientated by accommodating the model in a class as well as the preprocessor
- Added a script to split the dataset randomly into a training, validation and test set based on percentages rather than providing specific numbers
- Added a function to randomly pop up the CT-scan images for either a *sick* or *healthy* person to get a better understanding of the data
- The possibility to switch between ```cpu``` and ```gpu```
- Timing function to time each training round
- Plot functionallity to plot ```cpu``` time vs ```gpu``` time and to plot accuracy after each epoch

**Steps** <br>
To keep the documentation comprehensive we refer to the [original source](https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet "original source") of this code for further explanation, and will emphasize the added functionalities as well as the speed up obtained by using the ```gpu``` instead of the ```cpu```.

------------


***Pre procesing*** (```preprocess.py```) <br>
We used the original pre-processing code but moved this to a class to be able to easily play with the parameters:

Parameter  | Explanation|Default
------------- | -------------| -------------
```-s```  | The pixels to which the image should be adjusted to | 50
```-c```  | Number of slices to normalize to|20
```-o```  | The location to which the output should be written|-
```-i```  | Path to the folder containing the images|-
Example:
```bash
python preprocess.py -s 50 -c 20 -o data\processed_data -i data\sample_images\ -m data\stage1_labels.csv
```

------------

***Data splitting*** (```splitter.py```) <br>
To be able to train and test the 3D CNN, the input data needs to be divided into a training, validation and test set.  The original code did this by passing the indices for each set separately. We added the functionality to do this percentage based.

Parameter  | explanation | default
------------- | -------------|-------------
```-d```  | The dataset that needs to be split|-
```-r```  | % of the data that needs to be used for training|-
```-v```  | % of the data that needs to be used for validation|-
```-t```  | % of the data that needs to be used for testing|-
```-o```  | output_folder|-
Example:
```bash
python splitter.py -d data\processed_data.npy -r 0.6 -t 0.2 -v 0.2 -o data/
```
This will also show the number of instances in each of the sets:
> [INFO] total data size: 19
[INFO] train size: 11
[INFO] validate size: 4

------------

***Model training*** (```cnn_train.py```) <br>
Training the model requires the training and validation set obtained from ```splitter.py``` (see above). 

Parameter  | Explanation|Default
------------- | -------------| -------------
```-s```  | The pixels to which the image should be adjusted to | 50
```-c```  | Number of slices to normalize to|20
```-a```  | The number of classes in the dataset|2
```-b```  | The batch size|10
```-k```  | The  keep rate|0.8
```-e```  | The  number of epochs|10
```-g```  | If the GPU should be used|True
```-n```  | The model name (for saving purposes)|model
```-r```  | Path to the training data|-
```-v```  | Path to the validation data|-
```-o```  | Path to which the model should be saved|-
Example:
```bash
cnn.py -r data/training.npy -v data/validation.npy -o data/
```
This will show the obtained accuracy after each epoch: <br>
>[INFO] Epoch 1 completed out of 10 loss: 8292800128.0
[INFO] Accuracy: 0.75

And after all epochs the overall accuracy, fitment percentages and the total training time: <br>
>[INFO] Finished Accuracy: 0.75
[INFO] fitment percent: 1.0
[INFO] runtime: 15.150243174234687

------------

***Model testing*** (```cnn_test.py```) <br>
Training the model requires the test set obtained from ```splitter.py``` (see above). 
The paramters `-s` , `-c`, `-a`, `-b`, `-e` , and `-n` should generally be the same as for training. Often the keep rate (`-k`) is varied. 

Parameter  | Explanation|Default
------------- | -------------| -------------
```-s```  | The pixels to which the image should be adjusted to | 50
```-c```  | Number of slices to normalize to|20
```-a```  | The number of classes in the dataset|2
```-b```  | The batch size|10
```-k```  | The  keep rate|0.8
```-e```  | The  number of epochs|10
```-g```  | If the GPU should be used|True
```-n```  | The model name (for saving purposes)|model
```-o```  | Output path (if export is preferred)|None
```-t```  | Path to the test data|-
```-f```  | Path to the folder containing the model|-


Example:
```bash
cnn_test.py -f data/ -t data/test.npy -k 1.0
```
##  Evaluation ##
**GPU vs CPU**
The training was performed for different epoch values (`1,5,15,20,25,30,35.40,45,50`) using the `cpu` and the `gpu`. The results were used as input for the `plots.R` scripts which produced the following plot:
![Alt text](images/gpu_vs_cpu.PNG?raw=true "Title")
Further we calculated the speed up obtained by utilizing  the `gpu`:

Epoch  | GPU |CPU|Speed up
------------- | -------------| -------------|-------------
1  | 6.007857 | 321.3197 |53.48324
5  | 9.138309 |381.9040 |41.79154
10  | 13.228123|454.7322| 34.37617
15 | 18.195777|529.2489|29.08636
20 | 23.074444| 607.9874  |26.34895
25  |  28.200325 | 678.3122 | 24.05335
30 | 33.673778 | 753.0017| 22.36166
35 | 39.658525|823.5775|20.76672
40  |44.755630|900.9654 |20.13077
45 | 49.822851|981.2035 |19.69385
50 | 50 54.874518|1061.6786|19.34739

The average speed up obtained by utilizing the `gpu` is **28x faster**


**Testing the classifier**
The original source code did not contain any testing code so we added this (see above). Besides providing the obtained accuracy this will also produce a list containing the patient id, the given class and the predicted class. For example when running the test code on the sample image test set this will give the following output:
ID  | Label|Predicted
------------- | -------------| -------------
0a0c32c9e08cc2ea76a71649de56be6d	  | 0 | 0
0a099f2549429d29b32f349e95fb2244	|0	|0
0d19f1c627df49eb223771c28548350e	|0	|0
0ddeb08e9c97227853422bd71a2a695e	|0	|0
In this example the achieved accuracy is `100%`  as the label is the same as predicted in every case.
The provided output format is easy to use as input for [R](https://www.r-project.org/) .

Using the **whole dataset** we get an average accuracy of  `77%` , while this seems pretty good simply predicting the majority class in the dataset will give `74%` accuracy as the dataset consists of 1035 non-cancer and 362 cancerous examples. But yeaah we made it 3% better ;).  The confusion matrix corresponding to the whole dataset is:


|  | Healthy |Sick
------------- | -------------|-------------
**Healthy**  | 907 |80
**Sick**|153 |38


As can be seen from this data we only classified 38 out of the 362 cancerous examples. We could only classify obivous cancerours example, as for example this one:

![Alt text](images/sick_patient.PNG?raw=true "Title")

We are not doctors but the second slice on the second row clearly contains some abnormality. In contrast we were not able to classify this example:

![Alt text](images/sick_patient_missed.PNG?raw=true "Title")

