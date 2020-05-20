"SSDC-DenseNet: A Cost-Effective End-to-End Spectral-Spatial Dual-Channel Dense Network or Hyperspectral Image Classification" (SSDC-DenseNet)
the description of files in this repository:
"input_data.py" code for load training and testing data
"data_producer.py"code for creating patches from input data
"config.py"code for setting network parameters for different datasets
"feed.ipynb" code for training and testing of the models
"SSDC_Densenet.py" code for our network architecture
"cnn.py resnet.py"code for CNN-4 and ResNet-4 architecture
"contextualcnn.py fdssc.py"code for corresponding network architecture
"model_test_3d.py"for replacing the 2D convolution operations in Block 2 with 3D ones
"count_param_and_flops.py decoder.py" code for evaluating metrics 

If you use this code, please cite our work: Yutong Bai, Qifan Zhang, Zexin Lu,  and Yi Zhang, Bai, Y., Q. Zhang, Z. Lu & Y. Zhang (2019) SSDC-DenseNet: A Cost-Effective End-to-End Spectral-Spatial Dual-Channel Dense Network for Hyperspectral Image Classification. Ieee Access, 7, 84876-84889.


Any questions about this code, please contact the author: ybai1920@163.com