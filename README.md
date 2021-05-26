# Sign-Language-Classification
This is a Deep Learning Model to classify signs of **Sign Language** between 24 classes using CNN by Tensorflow and Keras.

![Sign-Language](signs.png)


The dataset format is patterned to match closely with the classic MNIST. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.



## Data
The Link of dataset on Kaggle: [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist).

The kernal on Kaggle: [Sign-Language Classification CNN 99% Accuracy](https://www.kaggle.com/abdelrahmanzied/sign-language-classification-cnn-99-accuracy).




## Prerequisites
1. You should have python on your computer.
2. Install libraries:

    `pip install numpy`

    `pip install pandas`
    
    `pip install matplotlib`
    
    `pip install seaborn`

    `pip install warnings`
    
    `pip install scikit-learn`
    
    `pip install tensorlfow`
    
    
    If You want to install tensorflow GPU version you can check my article on Midium:

    [Install Tensorflow and Keras on GPU on Windows in 2021 using CUDA and cuDNN — All Errors are Fixed](https://abdelrahmanzied.medium.com/install-tensorflow-and-keras-on-gpu-on-windows-in-2021-using-cuda-and-cudnn-all-error-fixed-8a3967398eb7).

    If you have all prerequisites of tensorflow-gpu, just install using this command:

    `pip install --upgrade tensorflow-gpu==2.4.1`




## Usage
You can download the model "best_model.hdf5" file from [Kaggle](https://www.kaggle.com/abdelrahmanzied/sign-language-classification-cnn-99-accuracy) notebook and import it using Tensorflow.

`Best_Model = tf.keras.models.load_model('best_model.hdf5')`


## Enjoy!
