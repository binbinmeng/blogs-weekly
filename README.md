# blogs-weekly

## main contents
1. object detection
  - faster R-CNN [一文读懂faster R-CNN](https://zhuanlan.zhihu.com/p/31426458)
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
  - [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199)
  - [pixor](http://www.cs.toronto.edu/~byang/projects/pixor/pixor_poster.pdf)

2. object tracking
  - template matching
  - [kalman filter](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits) [浅谈协方差矩阵](http://www.cnblogs.com/chaosimple/p/3182157.html) [good python jupyternotebook](https://github.com/NirvanaZhou/Kalman-and-Bayesian-Filters-in-Python)
  - KCF
  - LK optical flow
  - [multiple object tracking lidar](https://github.com/praveen-palanisamy/multiple-object-tracking-lidar/issues)
  - [sort](https://github.com/abewley/sort)
  - [deeprot](https://github.com/nwojke/deep_sort)


3. semantic segmentation
  - segnet
  - [enet](https://github.com/TimoSaemann/ENet/tree/master/Tutorial)
  - [erfnet](https://github.com/Eromera/erfnet_pytorch)

4. multi-task
  - blitzNet
  - [PANet](https://blog.csdn.net/u011974639/article/details/79595179)
  - maskRCNN
  
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
  - [quatization(8bit](https://zhuanlan.zhihu.com/p/42811261) 4bit xnor) and fix-point, [caffe int8 convert tools](https://github.com/BUG1989/caffe-int8-convert-tools), [google gemmlowp](https://github.com/google/gemmlowp/blob/master/doc/low-precision.md)
  - low rank decomposition
  - tucker tensor decomposation
  - [TensorRT](https://github.com/chenzhi1992/TensorRT-SSD)
  - [some opensource projects](https://blog.csdn.net/zhangjunhit/article/details/78901976)
  - [Ristretto: Hardware-Oriented Approximation of Convolutional Neural Networks](https://arxiv.org/pdf/1605.06402.pdf) the [website](http://ristretto.lepsucd.com/) and [code](https://github.com/pmgysel/caffe) and [colleague's note](https://blog.csdn.net/yingpeng_zhong/article/details/78232693) and [this excellent note](https://blog.csdn.net/xiaoxiaowenqiang/article/details/81713131)
  - [standford cs217](https://cs217.stanford.edu/)
  - [distill and quantization](https://github.com/antspy/quantized_distillation)

7. deep learning tricks
  - ohem
  - 1x1 convolution
  - depth wise convolution
  - group convolution
  - deformable convolution
  - dilation
  - deconvolution
  - roi pooling
  - [roi align](https://blog.csdn.net/u011918382/article/details/79455407)
  - [RPN](https://blog.csdn.net/u014380165/article/details/80380669)
  - optimization method(SGD,Adam,RMSProp)
  - [anchor](https://blog.csdn.net/u014380165/article/details/80379812)
  - 3D Convolution
  - [affine transform](https://www.matongxue.com/madocs/244.html), the classic y = Wx + b
  - [focal loss](https://blog.csdn.net/qq_34564947/article/details/77200104)
  - [softNMS](https://www.cnblogs.com/zf-blog/p/8532228.html)
  - [group normalization](http://www.dataguru.cn/article-13318-1.html)
  
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
  - [RANSAC](https://blog.csdn.net/zinnc/article/details/52319716) [MRPT中较好的实现](https://github.com/zhouxiaofan/mrpt/blob/master/libs/math/src/ransac.cpp)
  - [图像拼接](https://www.zhihu.com/question/20512919/answer/24912005)
  - [GPU accelerated Natural Image stitching](https://github.com/yhmtsai/GPU-accelerated-Natural-Image-Stitching-with-Global-Similarity-Prior)
  - [awesome image stitching](https://github.com/amusi/awesome-image-stitching)
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
  - [深度学习面试知识点](https://github.com/NirvanaZhou/Algorithm_Interview_Notes-Chinese)
16. non-linear optimization
  - [cerea solver](https://blog.csdn.net/wzheng92/article/details/79634069) (easier to use than g2o)
  
17. clustering algorithm
 - [DBScan](https://www.cnblogs.com/pinard/p/6208966.html)
 - [HAC](https://blog.csdn.net/eternity1118_/article/details/51520164)
 - [常见的6大聚类算法](https://blog.csdn.net/Katherine_hsr/article/details/79382249)
 
18. PCL
 - road map: [API Documentation](http://www.pointclouds.org/documentation/) -> [Tutorials](http://www.pointclouds.org/documentation/tutorials/) -> [Media](http://www.pointclouds.org/media/) -> [github/pcl](https://github.com/PointCloudLibrary/pcl) -> Blog -> User forum
 - [pcl segmentation and classification algorithms](https://blog.csdn.net/xiaoxiaowenqiang/article/details/79873816)
 - [RoN](https://www.cnblogs.com/ironstark/p/5010771.html)
 - [点云配准ICP等以及拼接](https://blog.csdn.net/Ha_ku/article/details/79755623)
 
19. programming
 - [code reading tools: graphviz+codeviz](https://abcdxyzk.github.io/blog/2016/03/21/graphviz-codeviz/)
 - [opencv](https://www.w3cschool.cn/opencv/opencv-2gnx28u3.html) and another [good tutorial](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/table_of_content_imgproc/table_of_content_imgproc.html#table-of-content-imgproc)
 - [c++](https://www.w3cschool.cn/cplusplus/) another [good resource](https://github.com/jwasham/practice-cpp) and [c++学习五步法](https://www.zhihu.com/question/20410487) [好的c++资源汇总](https://github.com/amusi/cpp-from-zero-to-one)
 - [c](https://github.com/jwasham/practice-c)
 - [python](https://github.com/jwasham/practice-python)
 - [关于python的面试题](https://github.com/taizilongxu/interview_python)
 
 
20. road map
 - [road map to SDE](https://github.com/jwasham/coding-interview-university/blob/master/translations/README-cn.md)
 - [slam求职建议](https://zhuanlan.zhihu.com/p/28565563)
 - [一篇讲得比较清楚的slam综述](https://zhuanlan.zhihu.com/p/23247395)
 
21. SLAM
 - [SLAM for dummies](https://zhuanlan.zhihu.com/p/32937247)
 
22. lidar
 - [apollo perception](https://zhuanlan.zhihu.com/p/33416142)
 - [apollo cnn seg](https://zhuanlan.zhihu.com/p/35034215)
 - [good resources](https://github.com/beedotkiran/Lidar_For_AD_references)
 
23. opencv
 - [opencv python tutorial](https://github.com/NirvanaZhou/Learning_OpenCV/tree/master/Python)
