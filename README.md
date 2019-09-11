# Learning to Disentangle Latent Physical Factors for Video Prediction


This repository contains datasets, code for dataset initialization and MIG 
evaluation scripts corresponding to:

D. Zhu, M. Munderloh, B. Rosenhahn, J. Stückler.
 **Learning to Disentangle Latent Physical Factors for Video Prediction.**
German Conference on Pattern Recognition (GCPR) 2019.

A video demonstrating the results can be found [here](https://m.youtube.com/watch?v=PZ9D4pqhkxs)

## Datasets Description
Three video datasets describing physical scenarios. Each sequence in 
these datasets has 10 frames in 1 second. Resolution is 128x128.

### Sliding Set
<img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/sliding/1.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/sliding/2.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/sliding/3.gif" height="96">

- Objects sliding on a plane
- Varying discrete shape, scale, friction, speed and position
- 26000 sequences with 20000/3000/3000 for training, validation, and test.

### Wall Set
<img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/wall/1.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/wall/2.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/wall/3.gif" height="96">

- Objects sliding into a wall
- Varying discrete shape, scale, material (density, restitution, friction, color), 
initial speed and position
- 10125 sequences with 7425/1350/1350 for training, validation, and test.

### Collision Set
<img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/collision/1.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/collision/2.gif" height="96"> <img src="https://github.com/TsuTikgiau/DisentPhys4VidPredict/blob/master/.README/collision/3.gif" height="96">

- Two objects sliding into each other
- Varying discrete shape, scale, material (density, restitution, friction, color), 
initial speed and position
- 30000 sequences with 25000/2500/2500 for training, validation, and test.




## How to Use
Datasets can be downloaded here: [Datasets.zip](https://owncloud.tuebingen.mpg.de/index.php/s/RXQKTQ9PdyrQwPq) (md5sum: 27ca28c4646c4fa77911338061f0c820)

Data are in the '.tfrecord' form. The code to load datasets can be found in the 
folder 'video_prediction/datastes'. The file 'scripts/eval_mig.py' demonstrates
how to initialize these datasets. Besides, it is also our implementation for 
Mutual Information Gap evaluation. TensorFlow version is v1.12.

Our code is based on Alex X. Lee's [SAVP](https://github.com/alexlee-gk/video_prediction)
and Ricky Tian Qi Chen's [beta-TCVAE](https://github.com/rtqichen/beta-tcvae). 
Their License can also be found in the license file.



## Citation
If you find this useful for your research, please cite the following:

    @article{Deyao2019GCPR,
        author    = {Deyao Zhu and Marco Munderloh and Bodo Rosenhahn and Jörg Stückler},
        title     = {Learning to Disentangle Latent  Physical Factors for Video Prediction},
        journal   = {German Conference on Pattern Recognition (GCPR)},
        year      = {2019},
    }




