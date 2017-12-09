# [Deep Complex Networks](https://arxiv.org/abs/1705.09792) on TensorFlow & Keras
  - Differences from [original Theano version](https://github.com/ChihebTrabelsi/deep_complex_networks)
    - Run on Tensorflow & Keras
    - Add a few new types of model
    - Generate TensorBoard-readable data
    - FFT, IFFT are disabled.
  - Notes
    - Make sure '~/.keras/keras.json' has
        ```
        "image_data_format": "channels_first", 
        "backend": "tensorflow"
        ```
    - Dependencies
        ```
        pip install numpy tensorflow-gpu keras kerosene
        pip install pydot graphviz (if you want a picture of model graph)
        ```
  - [README.md](https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/README.md) from Theano version for more detail (The follwing is the copy at this point of time.)
------------------------------    
Deep Complex Networks
=====================

This repository contains code which reproduces experiments presented in
the paper [Deep Complex Networks](https://arxiv.org/abs/1705.09792).

Requirements
------------

Install requirements for computer vision experiments with pip:
```
pip install numpy Theano keras kerosene
```

And for music experiments:
```
pip install scipy sklearn intervaltree resampy
pip install git+git://github.com/bartvm/mimir.git
```

Depending on your Python installation you might want to use anaconda or other tools.


Installation
------------

```
python setup.py install
```

Experiments
-----------

### Computer vision

1. Get help:

    ```
    python scripts/run.py train --help
    ```

2. Run models:

    ```
    python scripts/run.py train -w WORKDIR --model {real,complex} --sf STARTFILTER --nb NUMBEROFBLOCKSPERSTAGE
    ```

    Other arguments may be added as well; Refer to run.py train --help for
    
      - Optimizer settings
      - Dropout rate
      - Clipping
      - ...


### MusicNet

0. Download the dataset from [the official page](https://homes.cs.washington.edu/~thickstn/musicnet.html)

    ```
    mkdir data/
    wget https://homes.cs.washington.edu/~thickstn/media/musicnet.npz -P data/
    ```

1. Resample the dataset with 

    ```
    resample.py data/musicnet.npz data/musicnet_11khz.npz 44100 11000
    ```

2. Run shallow models

    ```
    train.py shallow_model --in-memory --model=shallow_convnet --local-data data/musicnet_11khz.npz
    train.py shallow__complex_model --in-memory --model=complex_shallow_convnet --complex --local-data data/musicnet_11khz.npz
    ```

3. Run deep models

    ```
    train.py deep_model --in-memory --model=deep_convnet --fourier --local-data data/musicnet_11khz.npz
    train.py deep_complex_model --in-memory --model=complex_deep_convnet --fourier --complex --local-data data/musicnet_11khz.npz
    ```

4. Visualize with jupyter notebook

    Run the notebook `notebooks/visualize_musicnet.ipynb`.

    ![precision-recall](imgs/precision_recall.png "Precision-recall curve")
    ![predicitons](imgs/pred_gt.png "Prediction example")


Citation
--------

Please cite our work as 

```
@ARTICLE {,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
```
