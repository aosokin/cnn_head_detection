# Context-aware CNNs for person head detection

Created by Anton Osokin and Tuan-Hung Vu at INRIA, Paris.

### Introduction

Person detection is a key problem for many computer vision tasks. While face detection has reached maturity, detecting people under a full variation of camera view-points, human poses, lighting conditions and occlusions is still a difficult challenge. In this work we focus on detecting human heads in natural scenes. Starting from the recent local R-CNN object detector, we extend it with two types of contextual cues. First, we leverage person-scene relations and propose a Global CNN model trained to predict positions and scales of heads directly from the full image. Second, we explicitly model pairwise relations among objects and train a Pairwise CNN model using a structured-output surrogate loss. The Local, Global and Pairwise models are combined into a joint CNN framework. To train and test our full model, we introduce a large dataset composed of 369,846 human heads annotated in 224,740 movie frames. We evaluate our method and demonstrate improvements of person head detection against several recent baselines in three datasets. We also show improvements of the detection speed provided by our model.

Our paper is available as [arXiv tech report](http://arxiv.org/abs/1511.07917). Our data and models are available on the [project web page](http://www.di.ens.fr/willow/research/headdetection).

### License

Our code is released under the MIT License (refer to the LICENSE file for details).

### Cite

If you find our code useful in your research, please, consider citing our paper:

>@inproceedings{vu15heads,<br>
    Author = {Vu, Tuan{-}Hung and Osokin, Anton and Laptev, Ivan},<br>
    Title = {Context-aware {CNNs} for person head detection},<br>
    Booktitle = {International Conference on Computer Vision ({ICCV})},<br>
    Year = {2015} }

### Contents
1. [Requirements](#requirements)
2. [Demo](#demo)
3. [Evaluation](#evaluation)
4. [Training](#training)
4. [Casablanca dataset](#casablanca-dataset)


### Requirements
To run the demo you just need MATLAB installed.

The full training/evaluation code requires [MatConvNet](http://www.vlfeat.org/matconvnet), CUDA, and a reasonable GPU.
We also recommend using [cuDNN](https://developer.nvidia.com/cudnn) for better performance.

The code was tested on Ubuntu 12.04 LTS with MATLAB-2014b, CUDA 7.0, cudnn-7.0-linux-x64-v3.0, and NVIDIA TITAN X.
We used [MatConvNet v1.0-beta12](https://github.com/vlfeat/matconvnet/archive/v1.0-beta12.zip).

Tested also with  cudnn-7.0-linux-x64-v4.0-rc and [MatConvNet v1.0-beta18](https://github.com/vlfeat/matconvnet/archive/v1.0-beta18.zip).

### Demo
[Demo](#demo) shows the precision-recall curves of our methods and main baselines on HollywoodHeads dataset.

1) Download the package and go to that folder
  ```Shell
  git clone https://github.com/aosokin/cnn_head_detection.git
  cd cnn_head_detection
  ```

2) Download and unpack the dataset
  ```Shell
  wget -P data http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip
  unzip data/HollywoodHeads.zip -d data
  ```

3) Download and unpack the detection results
  ```Shell
  wget http://www.di.ens.fr/willow/research/headdetection/release/results.zip
  unzip results.zip
  ```

4) Open MATLAB and run 
  ```Matlab
  demo
  ```

### Evaluation
[Evaluation](#evaluation) explains how to produce the detection results using the trained models. The results can be used to plot curves using [Demo](#demo).

0) To train the models you will need a descent GPU, CUDA and MatConvNet. We also recommend using [cuDNN](https://developer.nvidia.com/cudnn) for better performance. Let CUDAROOT and CUDNNROOT be the installation folders CUDA and cuDNN. Update your environment variables by, e.g., adding these lines to your .bashrc file:
  ```Shell
  export PATH=CUDAROOT/bin/:$PATH
  export LD_LIBRARY_PATH=CUDAROOT/lib64/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=CUDNNROOT/lib64/:$LD_LIBRARY_PATH
  ```
Installing MatConvNet is described [here](http://www.vlfeat.org/matconvnet/install). We compile the binaries by running the following commands from the root of MatConvNet (MATCONVNETROOT):
  ```Matlab
  cd matlab
  vl_setupnn
  vl_compilenn('enableGpu', true, 'cudaRoot', CUDAROOT, 'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', CUDNNROOT, 'enableImreadJpeg', true);
  ```

1) Download the package and go to that folder
  ```Shell
  git clone https://github.com/aosokin/cnn_head_detection.git
  cd cnn_head_detection
  ```

2) Compile the package and add the required paths. From MATLAB run
  ```Matlab
  compile_mex( CUDAROOT );
  setup( MATCONVNETROOT );
  ```

3) Download and unpack the dataset
  ```Shell
  wget -P data http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip
  unzip data/HollywoodHeads.zip -d data
  ```

4) Get the bounding-box proposals. If you want you can download ours computed with [Selective Search](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1):
  ```Shell
  wget -P data/HollywoodHeads http://www.di.ens.fr/willow/research/headdetection/release/candidates.zip
  unzip data/HollywoodHeads/candidates.zip -d data/HollywoodHeads
  ```

5) Get the models
  ```Shell
  wget http://www.di.ens.fr/willow/research/headdetection/release/models.zip
  unzip models.zip
  ```

6) You should be able to run these scripts from MATLAB command line:
  ```Matlab
  run_computeScores_localModel;
  run_computeScores_globalModel;
  ```

7) To compute scores of the pairwise model you need to compute the pairwise clusters. We have the precomputed version:
  ```Shell
  wget -P results/HollywoodHeads/pairwise http://www.di.ens.fr/willow/research/headdetection/release/imdb_pairwise_precomputedClusters.mat
  ```
Now you should be able to run this script from MATLAB command line (note, that you need the scores of the local model already computed, i.e. you need the result of run_computeScores_localModel.m):
  ```Matlab
  run_computeScores_pairwiseModel;
  ```

### Training
[Training](#training) explains how to train Local, Pairwise and Global models. The models can be used to produce results using [Evaluation](#evaluation) and [Demo](#demo).

0) Perform steps 1-4 of [Evaluation](#evaluation).

1) Get the pretrained model. You can get one from us:
  ```Shell
  wget -P models http://www.di.ens.fr/willow/research/headdetection/release/imagenet-torch-oquab.mat
  ```
Alternatively, you can get MatConvNet models trained on ImageNet [here](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification). We tested [imagenet-caffe-alex.mat](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat), [imagenet-vgg-s.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat), [imagenet-vgg-verydeep-16.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat).

2) Now you are ready to train the local and global models. For the local model launch the following in MATLAB:
  ```Matlab
  run_training_localModel
  ```
  For the global model do
  ```Matlab
  run_training_globalModel
  ```
The full training procedure requires several days of computation.

3) Training the pairwise model is sligthly more involved. First you need to have the local model trained and to compute its scores of all the candidates of the dataset. You can do this by running 
  ```Matlab
  run_computeScores_localModel;
  ```
with lines 11 and 16 changed to 
  ```Matlab
  resultFile = fullfile( resultPath, 'local', 'localModel-scores-test.mat' );resultFile = fullfile( resultPath, 'local', 'localModel-scores-test.mat' );
  scoreSubset = [1,2,3];
  ```
Running this procedure will require a lot of time.
Alternatively, you can download the scores we used.
  ```Shell
  wget -P results/HollywoodHeads/local http://www.di.ens.fr/willow/research/headdetection/release/localModel-scores-trainValTest.mat
  ```
Either way, you should be able to run 
  ```Matlab
  run_training_pairwiseModel
  ```

### Casablanca dataset
[Casablanca dataset](#casablanca-dataset) explains how to reproduce our results on the Casablanca dataset.
If you find the dataset useful in your research, please, cite the following papers:

>@inproceedings{ren08casablanca,<br>
    Author = {Ren, Xiaofeng},<br>
    Title = {Finding People in Archive Films through Tracking},<br>
    Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},<br>
    Year = {2008} }

1) Download and unpack the Casablanca dataset
  ```Shell
  wget -P data http://www.di.ens.fr/willow/research/headdetection/release/Casablanca.zip
  unzip data/Casablanca.zip -d data
  ```

2) Get the bounding-box proposals. If you want you can download ours computed with [Selective Search](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1):
  ```Shell
  wget -P data/Casablanca http://www.di.ens.fr/willow/research/headdetection/release/candidates_Casablanca.zip
  unzip data/Casablanca/candidates_Casablanca.zip -d data/Casablanca
  ```

3) Download and unpack the detection results
  ```Shell
  wget http://www.di.ens.fr/willow/research/headdetection/release/results_Casablanca.zip
  unzip results_Casablanca.zip
  ```

4) Open MATLAB and run
  ```Matlab
  demo_Casablanca;
  ```

To recompute our detections on the Casablanca dataset you can do the following steps. You can skip steps 5 and 6 if you already run evaluation for the HollywoodHeads dataset.

5) Download the models trained on the HollywoodHeads dataset and data for the pairwise clusters
  ```Shell
  wget http://www.di.ens.fr/willow/research/headdetection/release/models.zip
  unzip models.zip
  wget -P results/Casablanca/pairwise http://www.di.ens.fr/willow/research/headdetection/release/imdb_pairwise_precomputedClusters.mat
  ```

6) Compile the package and add the required paths. From MATLAB run
  ```Matlab
  compile_mex( CUDAROOT );
  setup( MATCONVNETROOT );
  ```

7) From MATLAB run
  ```Matlab
  run_computeScores_localModel_Casablanca;
  run_computeScores_globalModel_Casablanca;
  run_computeScores_pairwiseModel_Casablanca;
  ```
