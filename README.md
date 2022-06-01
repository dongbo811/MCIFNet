# MCIFNet_COD

Camouflaged object detection (COD), which aims to identify the objects that conceal themselves into the surroundings, has recently drawn increasing research efforts in the field of computer vision. In practice, the success of deep learning based COD is mainly determined by two key factors, including (i) A significantly large receptive field, which provides rich context information, and (ii) An effective fusion strategy, which aggregates the rich multi-level features for accurate COD. Motivated by these observations, in this paper, we propose a novel deep learning based COD approach, which integrates the large receptive field and effective feature fusion into a unified framework. Specifically, we first extract multi-level features from a backbone network. The resulting features are then fed to the proposed dual-branch mixture convolution modules, each of which utilizes multiple asymmetric convolutional layers and two dilated convolutional layers to extract rich context features from a large receptive field. Finally, we fuse the features using specially-designed multi-level interactive fusion modules, each of which employs an attention mechanism along with feature interaction for effective feature fusion. Our method detects camouflaged objects with an effective fusion strategy, which aggregates the rich context information from a large receptive field. All of these designs meet the requirements of COD well, allowing the accurate detection of camouflaged objects. Extensive experiments on widely-used benchmark datasets demonstrate that our method is capable of accurately detecting camouflaged objects and outperforms the state-of-the-art methods.


## Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single V100 GPU of 32 GB Memory.

> Note that our model also supports low memory GPU, which means you should lower the batch size.

1. Prerequisites:
   
    Note that SINet-V2 is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems (i.e., Windows) as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n SINet python=3.6`.
    
    + Installing necessary packages: [PyTorch > 1.1](https://pytorch.org/), [opencv-python](https://pypi.org/project/opencv-python/)

1. Prepare the data:

    + downloading testing dataset and move it into `./Dataset/TestDataset/`, 
    which can be found in [Google Drive](https://drive.google.com/file/d/120wKRvwXpqqeEejw60lYsEyZ4SOicR3M/view?usp=sharing).

    + downloading training/validation dataset and move it into `./Dataset/TrainValDataset/`, 
    which can be found in [Google Drive](https://drive.google.com/file/d/1bTIb2qo7WXfyLgCn43Pz0ZDQ4XceO9dE/view?usp=sharing)
    
    + downloading pretrained weights and move it into `./snapshot/SINet_V2/Net_epoch_best.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1XrUOmgB86L84JefoNq0gq2scBZjGaTkm/view?usp=sharing).
    
    + downloading Res2Net weights on ImageNet dataset [download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain_Val.py`.
    
    + Just enjoy it via run `python train.py` in your terminal.

2. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!


## Pre-computed maps: 
They can be found in download link([MCIF](https://pan.baidu.com/s/1CO_-VJyXwZhX894sf5xVtw)) on four testing dataset, including CHAMELEON, CAMO, COD10K, NC4K.


If you want to improve the usability or any piece of advice, please feel free to contact me directly (bodong.cv@gmail.com).
