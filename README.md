[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select aind-vui kernel"

## Project Overview

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

## Project Instructions

### Getting Started

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/udacity/AIND-VUI-Capstone.git
cd AIND-VUI-Capstone
```

2. Create (and activate) a new environment with Python 3.6 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name aind-vui python=3.5 numpy
	source activate aind-vui
	```
	- __Windows__: 
	```
	conda create --name aind-vui python=3.5 numpy scipy
	activate aind-vui
	```

3. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using the Udacity AMI, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==1.1.0
	```

4. Install a few pip packages.
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

6. Obtain the `libav` package.
	- __Linux__: `sudo apt-get install libav-tools`
	- __Mac__: `brew install libav`
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the `AIND-VUI-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

9. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `aind-vui` environment.  Open the notebook.
```
python -m ipykernel install --user --name aind-vui --display-name "aind-vui"
jupyter notebook vui_notebook.ipynb
```

10. Before running code, change the kernel to match the `aind-vui` environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select aind-vui kernel][image2]

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__


### Amazon Web Services

If you do not have access to a local GPU, you could use Amazon Web Services to launch an EC2 GPU instance.  Please refer to the [Udacity instructions](https://classroom.udacity.com/nanodegrees/nd889/parts/16cf5df5-73f0-4afa-93a9-de5974257236/modules/53b2a19e-4e29-4ae7-aaf2-33d195dbdeba/lessons/2df3b94c-4f09-476a-8397-e8841b147f84/project) for setting up a GPU instance for this project.


### Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


### Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `vui_notebook.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- The `sample_models.py` file with all model architectures that were trained in the project Jupyter notebook.
- The `results/` folder containing all HDF5 and pickle files corresponding to trained models.

Alternatively, your submission could consist of the GitHub link to your repository.


<a id='rubric'></a>
## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files      | The submission includes all required files.		|

#### STEP 2: Model 0: RNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Trained Model 0         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_0.pickle` are undefined.  The trained weights for the model specified in `simple_rnn_model` are stored in `model_0.h5`.   	|

#### STEP 2: Model 1: RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `rnn_model` module containing the correct architecture.   	|
| Trained Model 1         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_1.pickle` are undefined.  The trained weights for the model specified in `rnn_model` are stored in `model_1.h5`.   	|

#### STEP 2: Model 2: CNN + RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `cnn_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `cnn_rnn_model` module containing the correct architecture.   	|
| Trained Model 2         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_2.pickle` are undefined.  The trained weights for the model specified in `cnn_rnn_model` are stored in `model_2.h5`.   	|

#### STEP 2: Model 3: Deeper RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `deep_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `deep_rnn_model` module containing the correct architecture.   	|
| Trained Model 3         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_3.pickle` are undefined.  The trained weights for the model specified in `deep_rnn_model` are stored in `model_3.h5`.   	|

#### STEP 2: Model 4: Bidirectional RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `bidirectional_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `bidirectional_rnn_model` module containing the correct architecture.   	|
| Trained Model 4         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_4.pickle` are undefined.  The trained weights for the model specified in `bidirectional_rnn_model` are stored in `model_4.h5`.   	|

#### STEP 2: Compare the Models

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Question 1         		| The submission includes a detailed analysis of why different models might perform better than others.   	|

#### STEP 2: Final Model

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `final_model` Module         		| The submission includes a `sample_models.py` file with a completed `final_model` module containing a final architecture that is not identical to any of the previous architectures.   	|
| Trained Final Model        		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_end.pickle` are undefined.  The trained weights for the model specified in `final_model` are stored in `model_end.h5`.   	|
| Question 2         		| The submission includes a detailed description of how the final model architecture was designed.   	|


## Suggestions to Make your Project Stand Out!

#### (1) Add a Language Model to the Decoder

The performance of the decoding step can be greatly enhanced by incorporating a language model.  Build your own language model from scratch, or leverage a repository or toolkit that you find online to improve your predictions.

#### (2) Train on Bigger Data

In the project, you used some of the smaller downloads from the LibriSpeech corpus.  Try training your model on some larger datasets - instead of using `dev-clean.tar.gz`, download one of the larger training sets on the [website](http://www.openslr.org/12/).

#### (3) Try out Different Audio Features

In this project, you had the choice to use _either_ spectrogram or MFCC features.  Take the time to test the performance of _both_ of these features.  For a special challenge, train a network that uses raw audio waveforms!

## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
