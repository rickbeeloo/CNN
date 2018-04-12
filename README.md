Lung cancer detection using a 3D CNN 
-------------
*Authors: Rick Beeloo, Thomas Reinders and Koen van der Heide*

###Introduction###
Lung cancer is the leading cause of cancer-related deaths for both men and women across the developed world. Despite tremendous efforts to treat this cancer, the overall 5-year survival for all stages is dismally low at 15% [[1](https://www.ncbi.nlm.nih.gov/pubmed/9187198)]. In the daily practice of radiology, medical images from different modalities are read and interpreted by radiologists. Usually radiologists must analyze and evaluate these images comprehensively in a short time. But with the advances in modern medical technologies, the amount of imaging data is rapidly increasing. For example, CT examinations are being performed with thinner slices than in the past. The reading and interpretation time of radiologists will mount as the number of CT slices grows. Machine learning provides an effective way to automate the analysis and diagnosis for medical images. It can potentially reduce the burden on radiologists in the practice of radiology[[2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3372692/ "2")]. Although automatic cancer detection is faster than human curation, sequential machine learning is still time consuming and computer costly.  We therefore utilize graphics processing units (GPUs) to speed up this time consuming process and quantitively asses its profit. 



###Data###
For this project we used the [Sience Bowl](https://en.wikipedia.org/wiki/National_Science_Bowl "sience bowl") lung cancer data, which is available [here](https://www.kaggle.com/c/data-science-bowl-2017/data "here"):
- `stage1.7z` - contains all CT images for the first stage of the competition
- `stage1_labels.csv` - contains the cancer ground truth for the stage 1 training set images

These images are CT scans and thus comprise different slices:

###Code###
**Source and adjustments**
We adjusted [this code](https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet/notebook "this code") from Sentdex and therefore credits to him.  
The adjustments made:
- Wrote the code Object Orientated by accommodating the model in a class as wel as the preprocessor
- Added a script to split the dataset randomly into a training, validation and test set based on percentages rather than providing specific numbers
- Added a function to randomly pop up the CT-scan images for either a *sick* or *healthy* person to get a better understanding of the data
- The possibility to switch between ```cpu``` and ```gpu```. 
- Timing function to time each training round
- Plot functionallity to plot ```cpu``` time vs ```gpu``` time and to plot accuracy after each epoch

**Steps**
To keep the documentation comprehensive we refer to the [original source](https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet "original source") of this code for further explanation, and will emphasize the added functionalities as well as the speed up obtained by using the ```gpu``` instead of the ```cpu```.

------------
<Br>

***Pre procesing*** (```preprocess.py```)
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
<br>

***Data splitting*** (```splitter.py```)
To be able to train and test the 3D CNN, the input data needs to be devided into a training, validation and test set.  The original code did this by passing the indices for each set seperately. We added the functionality to do this percentage based.

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
<br>

***Model training*** (```cnn.py```)
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
```-t```  | Path to the training data|-
```-v```  | Path to the validation data|-
```-o```  | Path to which the model should be saved|-
Example:
```bash
cnn.py -t data/training.npy -v data/validation.npy -o data/
```
This will show the obtained accuracy after each epoch:
>[INFO] Epoch 1 completed out of 10 loss: 8292800128.0
[INFO] Accuracy: 0.75

And after all epochs the overall accuracy, fitment precentages and the total training time:
>[INFO] Finished Accuracy: 0.75
[INFO] fitment percent: 1.0
[INFO] runtime: 15.150243174234687
