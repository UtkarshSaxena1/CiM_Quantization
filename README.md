# CiM_Quantization
This repository is the official implementation of the paper :  [Partial-Sum Quantization for Near ADC-Less Compute-In-Memory Accelerators](https://ieeexplore.ieee.org/abstract/document/10244291) which was presented at ISLPED 2023.

## Versions
Pytorch version: 1.10.2 <br />
CUDA Version: 12.0 <br />
check environment.yml for more info. <br />

## Running the code
Steps to run the code:
1. Setup the environment using the environment file. Make sure to correct the prefix directory in the environment.yml file.

   ``` conda env create -f environment.yml ```
2. To run training on CIFAR-10:

   ```
   source setup.sh
   python ./examples/classifier_cifar10/main_lsq.py --hp ./examples/classifier_cifar10/prototxt/resnet_w3a3.prototxt
   ```

## Notes
1. The specifications about the training run are provided using prototxt file located at ./examples/classifier_cifar10/prototxt/resnet_w3a3.prototxt
2. Details about the CiM aware quantization can be found at ./models/_modules/lsq.py
3. Quantization function for weight and activation quantization is taken from lsq (https://arxiv.org/abs/1902.08153) paper.


## References
If you use this code, please cite the following paper:
```
@INPROCEEDINGS{10244291,
  author={Saxena, Utkarsh and Roy, Kaushik},
  booktitle={2023 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED)}, 
  title={Partial-Sum Quantization for Near ADC-Less Compute-In-Memory Accelerators}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ISLPED58423.2023.10244291}}
```
```
