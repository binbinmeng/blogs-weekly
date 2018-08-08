# blogs-weekly

## main contents
1. object detection
  - [RRPN](https://github.com/mjq11302010044/RRPN)
  - rfcn
  - ssd/rrc
  - yolo v1 v2 v3
  - fpn/ retina-net
  - refineDet
  - voxelnet
  - pointnet/pointnet++/f-pointnet
  - [DetNet](https://zhuanlan.zhihu.com/p/39702482)
  - [CFENet](https://blog.csdn.net/nwu_NBL/article/details/81087567)

2. object tracking
  - template matching
  - calman filter 
  - KCF
  - LK optical flow
  - [multiple object tracking lidar](https://github.com/praveen-palanisamy/multiple-object-tracking-lidar/issues)


3. semantic segmentation
  - segnet
  - enet
  - erfnet

4. multi-task
  - blitzNet
  
5. classification
  - ResNext
  - DenseNet
  - [DPN(dual path network) ](https://blog.csdn.net/u014380165/article/details/75676216)
  - edlenet
  - resnet (the recent paper about its essence)
  - mobilenet v1 v2
  - squeezenet
  
6. model optimization & acceleration
  - depth-wise convolution
  - group convolution
  - [network slimming](https://blog.csdn.net/u014380165/article/details/79969132)
  - [channel pruning](https://blog.csdn.net/u014380165/article/details/79811779) and [this original paper](https://github.com/yihui-he/channel-pruning)
  - [distill](https://github.com/NervanaSystems/distiller)
  - winograd
  - quatization(8bit 4bit xnor) and fix-point
  - low rank decomposition
  - tucker tensor decomposation
  - [TensorRT](https://github.com/chenzhi1992/TensorRT-SSD)
  - [some opensource projects](https://blog.csdn.net/zhangjunhit/article/details/78901976)

7. deep learning tricks
  - ohem
  - 1x1 convolution
  - depth wise convolution
  - group convolution
  - deformable convolution
  - dilation
  - deconvolution
  - roi pooling
  - [RPN](https://blog.csdn.net/u014380165/article/details/80380669)
  - optimization method(SGD,Adam,RMSProp)
  - [anchor](https://blog.csdn.net/u014380165/article/details/80379812)
  - 3D Convolution
  - [affine transform](https://www.matongxue.com/madocs/244.html), the classic y = Wx + b
  
 8. projects
  - cross walk detection
  - lane detection   [pls refer to hzh's work](https://zhouxiaofan.github.io/)
  - multi-object tracking(initialize with IDs and locations, track all objects)
  - [orb slam](https://blog.csdn.net/u010128736/article/list/1)
  - edlenet c++ implementation
  - tensorboard visualize all training process [1](http://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html#id1) [2](https://github.com/lanpa/tensorboardX)
  - pytorch study [marvan zhou's github](https://github.com/MorvanZhou/PyTorch-Tutorial) and [his website](https://morvanzhou.github.io/tutorials/machine-learning/torch/)
  - [system that contians object detection, tracking, segmentation](https://github.com/GustavZ/realtime_object_detection)

 9. image stitching
  - [RANSAC](https://blog.csdn.net/zinnc/article/details/52319716)
  - [图像拼接](https://www.zhihu.com/question/20512919/answer/24912005)
 10. visual odometry
 
 11. ROS(foundation of robot)
  - [apollo lidar](https://blog.csdn.net/qq_33801763/article/details/79092240)
  - [点云数据处理方法概述](http://www.p-chao.com/2017-06-11/%E7%82%B9%E4%BA%91%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95%E6%A6%82%E8%BF%B0/)
  - [点云数据处理学习笔记](https://blog.csdn.net/xs1997/article/details/78501120)
  - [基于点云的3D障碍物检测](https://blog.csdn.net/qq_33801763/article/details/79283017)
  - 鸟瞰图上物体检测与跟踪
  
 
 12. [Efficient Methods and Hardware for Deep Learning](https://platformlab.stanford.edu/Seminar%20Talks/retreat-2017/Song%20Han.pdf)
      also see [this](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf) and the [video](https://www.youtube.com/watch?v=eZdOkDtYMoo)  another course [Hardware Accelerators for Machine Learning](https://cs217.github.io/)
      
 13. 图像处理
 - [二值图像的连通域分析](https://blog.csdn.net/qq_37059483/article/details/78018539)
 - [ffmpeg](https://blog.csdn.net/leixiaohua1020/article/details/15811977)
 
 14. deep learning framework
 - pytorch [chenyun's pytorch tutorial](https://github.com/chenyuntc/pytorch-book) [tutorial 1](https://github.com/yunjey/pytorch-tutorial) [tutorial 2](https://github.com/MorvanZhou/PyTorch-Tutorial) [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
 - tensorflow [tutorial 1](https://github.com/yunjey/davian-tensorflow)  [tutorial 2](https://github.com/MorvanZhou/Tensorflow-Tutorial) [tutorial 3](https://github.com/MorvanZhou/Tensorflow-Computer-Vision-Tutorial)

15. interview
  - [阿里图像算法工程师内推电话面试记录](https://www.nowcoder.com/discuss/88050)
  - [top 35 python interview questions](https://data-flair.training/blogs/top-python-interview-questions-answer/)
  - [python interview](https://github.com/taizilongxu/interview_python#1-python%E7%9A%84%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0%E4%BC%A0%E9%80%92)
  
16. non-linear optimization
  - [cerea solver](https://blog.csdn.net/wzheng92/article/details/79634069) (easier to use than g2o)
