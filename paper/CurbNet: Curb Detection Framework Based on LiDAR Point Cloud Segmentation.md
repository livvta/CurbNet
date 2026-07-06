## Page 1

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
1
CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation
Guoyang Zhao, Fulong Ma, Weiqing Qi, Yuxuan Liu, Ming Liu, and Jun Ma, Senior Member, IEEE
Abstract—Curb detection is a crucial function in intelligent driving, essential for determining drivable areas on the road. However, the complexity of road environments makes curb detection challenging. This paper introduces CurbNet, a novel framework for curb detection utilizing point cloud segmentation. To address the lack of comprehensive curb datasets with 3D annotations, we have developed the 3D-Curb dataset based on SemanticKITTI, currently the largest and most diverse collection of curb point clouds. Recognizing that the primary characteristic of curbs is height variation, our approach leverages spatially rich 3D point clouds for training. To tackle the challenges posed by the uneven distribution of curb features on the xy-plane and their dependence on high-frequency features along the z-axis, we introduce the Multi-Scale and Channel Attention (MSCA) module, a customized solution designed to optimize detection performance. Additionally, we propose an adaptive weighted loss function group specifically formulated to counteract the imbalance in the distribution of curb point clouds relative to other categories. Extensive experiments conducted on 2 major datasets demonstrate that our method surpasses existing benchmarks set by leading curb detection and point cloud segmentation model. Through the post-processing refinement of the detection results, we have significantly reduced noise in curb detection, thereby improving precision by 4.5 points. Similarly, our tolerance experiments also achieve state-of-the-art results. Furthermore, real-world experiments and dataset analyses mutually validate each other, reinforcing CurbNet's superior detection capability and robust generalizability. The project website is available at: https://github.com/guoyangzhao/CurbNet/.
Index Terms—Point cloud, Curb detection, Segmentation, Deep learning, Autonomous driving.
I. INTRODUCTION
Autonomous vehicles fundamentally depend on analyzing data from onboard sensors to understand their surrounding environment, a cornerstone for safe driving [1], [2]. In this
This work was supported in part by the National Natural Science Foundation of China under Grant 62303390; in part by the Guangdong Provincial Key Lab of Integrated Communication, Sensing and Computation for Ubiquitous Internet of Things under Grant 2023B1212010007; in part by the Guangzhou HKUST/G2 Joint Funding Scheme under Grant 2024A03J0618. (Corresponding author: Jun Du.)
Guoyang Zhao, Fulong Ma, Weiqing Qi, and Ming Liu are with the Robotics and Autonomous Systems Trust, The Hong Kong University of Science and Technology (Guangzhou), Guangdong 51145138; Chin. (e-mail: gzhao492@connect.hkust-gz.edu.cn; fmaaf@connect.hkust-gz.edu.cn; wqiuad@connect.hkust-gz.edu.cn; celium@hkust-gz.edu.cn)
Yuxuan Liu is with the Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China (e-mail: yliuhb@connect.ust.hk)
Jun Ma is with the Robotics and Autonomous Systems Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511453, China (e-mail: 2018.2020.2019.2019.2019.2019.2019.2019.2019.2019.2019.2019.2019.2019.

---

## Page 2

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
2
vehicles [12].
Curb detection using LiDAR can be classified into manual feature methods and learning based methods  $[13]$ . Manual feature design methods  $[14]$ ,  $[15]$  typically analyze geometric relationships such as height and angle changes between adjacent points, given the difference between drivable roads and curbs  $[16]$ ,  $[17]$ . Most curb detection methods follow a sequential point cloud processing procedure  $[18]$ ,  $[19]$ , including stages of candidate region extraction, manual feature setting and clustering, and post-processing for fitting estimation. These methods, due to their interpretability in terms of safety, are widely used in curb detection for autonomous driving systems. However, the reliability of these manually designed rule-based processes is limited in practical applications, as errors in early detection stages can severely impact subsequent recognition performance. Additionally, they rely on manual feature design heavily, necessitating extensive parameter adjustments for various scenarios such as straight roads, curved roads, and different types of intersections, resulting in low practical efficiency and generalization  $[16]$ ,  $[20]$ .
With the breakthroughs in deep learning applications in perception, its capabilities in automatic feature extraction and learning have significantly reduced the tedium of manual feature design and parameter adjustment  $[21]$ ,  $[22]$ . Particularly, in addressing different road scenarios, it has substantially enhanced model recognition performance and robustness  $[23]$ . Recently, some researchers have explored curb detection using CNN-based methods, projecting 3D LiDAR point cloud data into 2D images from a Bird's Eye View (BEV) perspective, followed by processing these images with CNN models to detect curbs  $[24]$ ,  $[25]$ . However, this direct projection of 3D LiDAR data can lead to the loss of essential spatial structural information, particularly the crucial height difference information for curb detection  $[26]$ .
Considering current research in curb detection and the physical properties of 3D LiDAR, we have identified several challenges that need addressing, as shown in Fig. 1: (a) The primary distinguishing feature of curbs is the subtle height variation from the road surface [14], a challenging feature for models to accurately learn. (b) Curb point clouds from LiDAR scanning show significant distribution differences across various distances [27]. (c) Curbs occupy only a small portion of LiDAR point clouds, appearing as long and narrow lines, making effective model training difficult [26].
We first propose the 3D-Curb dataset, which contains 3D annotated point cloud data with a wide range of road scenes. Unlike most existing algorithms that project 3D point clouds onto 2D images, we extract features directly from point clouds, preserving crucial 3D spatial information. To address challenges (a) and (b), we propose a Multi-Scale and Channel Attention (MSCA) module. This module employs a multiscale fusion stage to mitigate the uneven distribution of curb point clouds and a channel attention stage to dynamically capture height variations along the z-axis feature. For challenge (c), which pertains to the severe imbalance in point cloud quantities among different categories, we introduce a novel loss combination. This includes an adaptive cross-entropy (ACE) loss and an IoU-focused loss, designed to
handle the imbalanced data distribution effectively. Moreover, recognizing the inherent sparsity of curb point clouds, we augment the training process by adding curb-related categories (such as roads and sidewalks) to assist in model learning. Finally, to further improve the precision of curb detection, we propose a post-processing scheme of multi-cluster and then fitting, which effectively removes the noise around the curb detection results.
Our primary contributions are summarized as follows:
1) We introduce a comprehensive 3D-Curb point cloud dataset based on SemanticKITTI, which is the largest and most diverse currently available to our knowledge.
2) We propose a novel Multi-Scale and Channel Attention (MSCA) module and an imbalance loss strategy, tailored to the distribution characteristics of curb point clouds.
3) We develop a multi-cluster fitting post-processing approach to further enhance detection performance.
4) Our method achieves state-of-the-art detection results in complex, large-scale intersection scenarios.
II. RELATED WORKS
A. Manual Feature Extraction Methods
In the early development of LiDAR technology, researchers designed mathematical functions to manually extract curb features by understanding the principles of LiDAR scanning. This process primarily involves stages of feature point extraction, feature point classification and filtering, and curb curve fitting and estimation. In [28], feature points are extracted through image segmentation and energy minimization, followed by the application of principal curves and surfaces methods in [29] for fitting detected curbs. Studies like [14], [15], [30], [31] utilize the horizontal and vertical continuity of point clouds, employing angle and height thresholds for curb feature extraction and using Gaussian Process Regression (GPR) and Random Sample Consensus for curb curve fitting. [15], [16] integrate the generalized curvature method from LOAM [32] into curb detection, refining the process with GPR. [16], [33] detect curbs by analyzing ring compression in dense 3D LiDAR data, employing false positive filters and least-squares regression filters based on height values, respectively. [19], [34] propose sliding-beam segmentation and sliding-window detection methods by analyzing individual LiDAR scan lines, focusing on specific curb detection in each frame.
However, these manual feature extraction and sequential processing methods are inefficient. Not only is feature creation laborious and requires specialized knowledge, but early-stage erroneous selections can impact later detection phases, making them inadequate for complex road scenes and diverse curb shapes  $[35]$ . In this context, deep learning methods can effectively solve the issues of feature extraction and generalization.
B. Deep Learning Methods
With the advancement of deep learning, some researchers have begun using CNNs to detect curbs in LiDAR point clouds, achieving notable performance. A common characteristic of deep learning-based curb detection methods is the

---

## Page 3

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
3
transformation of input 3D point cloud data into 2D view images or voxelization. [36] uses camera images, LiDAR, and elevation gradients of LiDAR as inputs, employing convolutional recurrent networks to extract road boundaries in 2D BEV images to construct semantic maps. [25] projects motion-accumulated 3D point cloud data onto 2D BEV images, initially detecting visible road edges through a U-Net network [37], followed by predicting obscured road boundaries using multi-layer convolutional networks with expanded receptive fields [38]. Similarly, [24] proposes a two-stage curb detection framework, initially employing U-Net for visible curb detection, then incorporating uncertainty quantification to improve detection performance in obscured areas.
Compared to traditional methods, these approaches demonstrate robustness in various driving environments and reduce the burden of manual parameter tuning. However, converting point clouds to 2D images results in significant loss of 3D information, especially the height difference features crucial for distinguishing curbs from other categories. [39] explores voxelizing the raw 3D point cloud as a superimage input to the model, using CNNs to detect road edges and lane markings, but its recognition accuracy remains low. LCDeT [26] employs a Transformer model for curb detection in voxelized point clouds, introducing dual attention mechanisms in both temporal and spatial dimensions to ensure detection stability and accuracy. Yet, this method relies on complex model structures and high-resolution LiDAR sensors.
Some general point cloud segmentation models [40]–[42] have achieved accurate 3D point cloud detection using relatively simple model structures. Among them, 3D U-Net [40] is a popular model for 3D volumetric data processing, adapted for point cloud segmentation by learning the spatial hierarchies of features. Cylinder3D [41] is a voxel-based method specifically designed to handle large-scale point clouds by projecting them into cylindrical grids, enabling more efficient feature extraction in 3D space. P.W.K [42] is an improvement of Cylinder3D based on knowledge distillation, focusing on transferring knowledge from a large teacher model to a smaller student network to enhance segmentation performance. However, these models are not specifically designed to extract curb features, which are crucial for curb detection tasks.
Our proposed method uses original point cloud as input, preserving more 3D feature information. Addressing the uneven distribution of curb point cloud feature and the easy loss of curb height difference information, we introduce a multi-scale and channel attention mechanisms to enhance performance.
C. Curb Detection Datasets
There is a significant amount of research in LiDAR-based curb detection; however, high-quality datasets with 3D annotations are scarce. Major automotive datasets such as NuScenes [43], KITTI [44], and SemanticKITTI [45] do not include curb annotations. The robustness of deep learning methods is closely related to the volume of data collected under various environmental conditions. These factors have somewhat hindered the application of deep learning methods in this task. [19] created a public dataset for curb detection, comprising
![](page_0002/images/0.jpg)

Fig. 2. 3D-Curb dataset construction process. Mainly developed based on the standard SemanticKITTI dataset.
TABLE I COMPARISON OF RELATED CURB DATASETS.
<table><tr><td>Dataset</td><td>LiDAR</td><td>Public</td><td>Frames</td><td>3D labeling</td><td>Categories</td></tr><tr><td>Zhang [19]</td><td>32-line</td><td>Yes</td><td>200</td><td>No</td><td>1</td></tr><tr><td>Liang [36]</td><td>-</td><td>No</td><td>-</td><td>No</td><td>1</td></tr><tr><td>Suleymanov [25]</td><td>-</td><td>No</td><td>-</td><td>No</td><td>1</td></tr><tr><td>Uncertainty [24]</td><td>32-line</td><td>Yes</td><td>5224</td><td>No</td><td>2</td></tr><tr><td>NRS [26]</td><td>128-line</td><td>Yes</td><td>6220</td><td>No</td><td>2</td></tr><tr><td>3D-Curb (ours)</td><td>64-line</td><td>Yes</td><td>7100</td><td>Yes</td><td>29</td></tr></table>
200 scans collected across five different scenes. [24] developed a public dataset for curb detection, including 5200 scans with BEV labels, collected from urban areas. LCDeT [26] introduced a curb dataset for 128-line LiDAR, containing 6200 frames of point clouds from urban road scenes, both during daytime and nighttime. [27] proposed a method for 3D curb detection and annotation in LiDAR point clouds, effectively reducing manual annotation time by $50\%$.
The above-mentioned curb point cloud datasets only label the curb category, which cannot be directly used in autonomous driving scenarios, as recognition of other categories such as vehicles and roads is also necessary. Similarly, other categories surrounding the curb line, like roads and sidewalks, can also assist the model in more accurately learning curb features. Based on the existing large SemanticKITTI dataset, we added annotations for the curb category, thereby covering a richer array of real road scenes, totaling up to 7100 frames of point cloud data. To our knowledge, this is currently the largest and most comprehensive curb point cloud dataset with annotations relevant to autonomous vehicles (AV). Table 1 illustrates the related LiDAR datasets for curb detection.
III. METHODOLOGY
A. 3D-Curb Dataset Construction
Compared to other autonomous driving scenarios, there is a significant lack of relevant curb point cloud datasets, especially those with 3D annotations. Building on the large-scale open-source SemanticKITTI dataset [45], we have developed and introduced the 3D-Curb dataset. This dataset retains the original 28 semantic categories while adding a new curb category. It was collected using a Velodyne HDL-64E LiDAR, providing comprehensive views of various street scenes as a general-purpose autonomous driving dataset.
The construction process of our dataset is illustrated in Fig. 2. Due to the high-frequency data acquisition of the SemanticKITTI dataset, the high similarity between frames can lead to model overfitting and significantly increase the

---

## Page 4

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
4
![](page_0003/images/0.jpg)

Fig. 3. Overview of proposed CurbNet framework. From left to right, first is point cloud data input and voxelization. Then there is a 5-layer deep encoder-decoder structure. Next comes the feature aggregation and segmentation head. Finally, the post-processing refinement of the detection results.
annotation workload. Therefore, we randomly selected 7100 representative frames from sequences 00-10. We utilized the high-quality road annotations provided by the SemanticKITTI dataset and applied the Ground Plane Fitting method proposed by [46] to extract the boundaries within the road labels. We then extended these parameters to obtain the curb labels. However, due to static/dynamic obstacles and other occlusions, these automatically generated curb annotations contained many inaccuracies. Thus, we manually refined the curb labels in the BEV perspective to ensure high-quality annotations. The 3D-Curb dataset focuses on curb annotations in the forward direction of vehicle travel, with an average range of $40.43$ meters along the forward y-axis. To accentuate the curb areas, the lateral x-axis range is set to 1.3 times the road width.
To the best of our knowledge, this is the largest curb point cloud dataset to date and the only one with 3D annotations. Table I compares our dataset with other related curb datasets.
B. Overview of Model Framework
As illustrated in Fig. 3, the CurbNet framework comprises four main components: point cloud input with voxelization, feature extraction, feature aggregation to segmentation, and post-processing refinement.
In most road scenarios, curbs are primarily located at the junction between the road and the sidewalk. To enhance the model's ability to learn curb features, we introduced additional labels for the road and sidewalk during the training process. To align the input point cloud data with physical space characteristics (where the density of circularly scanned point clouds decreases with increasing distance), we partition the voxel blocks based on the point cloud distance. This approach minimizes the impact of uneven point cloud distribution.
During the feature extraction stage, voxelized features with dimensions  $F \times H \times W \times D$  are fed into a 5-layer deep encoder-decoder structure. Each encoder and decoder backbone incorporates the MSCA module, specifically designed for curb point cloud feature extraction. A stride-2 sparse convolution is employed as the pooling function.
Subsequently, the high-dimensional voxelized features  $(F \times H \times W \times D)$  are aggregated and then input into the segmentation head to obtain the detection results. Finally, the curb detection results undergo post-processing involving multiclustering, fitting, and noise removal to further enhance detection accuracy.
The integration of these components within the CurbNet framework enables robust and precise curb detection, addressing the challenges posed by varying point cloud densities and the complex spatial characteristics of road environments.
C. Multi-Scale and Channel Attention (MSCA) Module
The structure of the MSCA module, illustrated in Fig. 4, serves as a fundamental component of the encoder-decoder architecture. The MSCA module is composed of two main parts: multi-scale fusion and channel attention.
Multi-Scale Fusion. Curbs in point clouds typically form a long curve along both sides of the road. However, as the LiDAR scanning distance increases, the point cloud density decreases, leading to sparser curb features. This results in significant scale differences in the features represented by the same number of point clouds at different distances after voxelization. To address this, we designed the MSCA module with a multi-scale fusion strategy. We employ convolution groups with different strides (1, 3, 5) to construct a feature pyramid, and then use dense pyramid connections to deeply fuse the output features of different scales. Inspired by text detection methods [47], we adopt three asymmetric sparse convolution kernels to target the regions in the xy, xz, and yz planes within the voxel space. This approach captures multi-dimensional curb feature information. Compared to traditional 3×3×3 convolutions, the combination of asymmetric sparse convolutions offers higher computational efficiency while maintaining the same receptive field.
Formally, given the input feature map  $X \in R^{F \times H \times W \times D}$ , where  $F \times H \times W \times D$  denotes the spatial dimensions, we apply convolutions with varying strides to capture multi-scale features. Specifically, the input is processed using sparse con-

---

## Page 5

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
5
volutions (denoted as SConv) with stride values $s \in \{1,3,5\}$:
$$
\mathbf {X} _ {s 1} = \operatorname{SConv} _ {s = 1} (\mathbf {X}),
$$
$$
\mathbf {X} _ {s 3} = \operatorname{SConv} _ {s = 3} (\mathbf {X}), \tag {1}
$$
$$
\mathbf {X} _ {s 5} = \operatorname{SConv} _ {s = 5} (\mathbf {X})
$$
These convolutions produce feature maps  $X_{s1}$ ,  $X_{s3}$ , and  $X_{s5}$  at different spatial scales. The multi-scale outputs are fused as follows:
$$
\mathbf {X} _ {\mathrm{ms}} = f (\mathbf {X} _ {s 1}, \mathbf {X} _ {s 3}, \mathbf {X} _ {s 5}) \tag {2}
$$
where  $f(\cdot)$  represents the pyramid fusion operation, typically achieved by concatenating the outputs followed by additional sparse convolutions for deep integration.
Channel Attention. The primary distinguishing feature of curbs is the subtle height difference between the road and the sidewalk, reflected in the z-axis of point cloud data. Unlike general point cloud segmentation algorithms such as Cylinder3D [41], which primarily focus on feature learning in the xy plane, we designed the channel attention module to capture high-frequency features along the z-axis. Initially, a  $1 \times 1 \times D$  sparse convolution is applied to preliminarily extract channel features:
$$
\mathbf {C} = \operatorname{SConv} _ {1 \times 1 \times D} (\mathbf {X}) \tag {3}
$$
The channel features are then processed through an encoder-decoder structured MLP to further refine the height feature:
$$
\mathbf {C} _ {\text { encoded }} = \mathrm{MLP} _ {\text { enc }} (\mathbf {C}), \quad \mathbf {C} _ {\text { decoded }} = \mathrm{MLP} _ {\text { dec }} (\mathbf {C} _ {\text { encoded }}) \tag {4}
$$
The output of the MLP,  $C_{ decoded}$ , is then passed through two parallel branches. In the first branch, a softmax function generates dynamic weights for each channel:
$$
\mathbf {W} _ {\text { channel }} = \operatorname{softmax} (\mathbf {C} _ {\text { decoded }}) \tag {5}
$$
In the second branch,  $C_{ decoded}$  is further processed by another  $1 \times 1 \times D$  sparse convolution:
$$
\mathbf {C} _ {\text { conv }} = \mathrm{SCConv} _ {1 \times 1 \times D} (\mathbf {C} _ {\text { decoded }}) \tag {6}
$$
The $\mathbf{W}_{\mathrm{channel}}$ are element-wise multiplied with the output $\mathbf{C}_{\mathrm{conv}}$ to produce the final channel-attended feature map:
$$
\mathbf {X} _ {\text { channel }} = \mathbf {W} _ {\text { channel }} \odot \mathbf {C} _ {\text { conv }} \tag {7}
$$
The final output of the MSCA is obtained by combining the multi-scale fusion output and the channel attention output:
$$
\mathbf {X} _ {\text { output }} = \mathbf {X} _ {\mathrm{ms}} + \mathbf {X} _ {\text { channel }} \tag {8}
$$
By combining multi-scale feature extraction with dynamic channel attention, the MSCA module effectively captures both spatial and height variations in point cloud data, thereby enhancing the model's ability to detect curbs.
![](page_0004/images/0.jpg)

Fig. 4. Structure of multi-scale and channel attention (MSCA) module. SConv means Sparse convolution layer. Multi-Scale Fusion is mainly used to fuse spatial features of different scales, and Channel Attention is used to dynamically extract height features of the z-axis.
D. Loss Group
In real-world scenarios, the point cloud data for curbs comprises only a small fraction compared to other categories such as roads and buildings. Using uniform loss weights can lead to training imbalances, adversely affecting the recognition performance for the minority class, i.e., curbs. To address this, we propose a novel combination of Adaptive Cross-Entropy (ACE) Loss and Lovász-Softmax Loss.
I) Adaptive Cross Entropy (ACE) Loss
Due to the imbalance between the number of curb point clouds and the number of other categories such as roads and buildings. Standard loss functions like Cross Entropy (CE) [48] do not adequately address this imbalance, leading to suboptimal performance in recognizing minority classes. The standard CE loss is defined as:
$$
\mathcal {L} _ {C E} \left(p _ {\mathrm{t}}\right) = - \log \left(p _ {\mathrm{t}}\right) \tag {9}
$$
where $p_t$ represents the predicted probability of the true class.
Given the disproportionate representation of classes in the point cloud data, we draw inspiration from the Focal Loss [49] to reallocate the loss contribution of easy and hard samples, significantly reducing the influence of the majority background samples:
$$
\mathcal {L} _ {F L} \left(p _ {\mathrm{t}}\right) = - \alpha_ {\mathrm{t}} \left(1 - p _ {\mathrm{t}}\right) ^ {\gamma} \log \left(p _ {\mathrm{t}}\right) \tag {10}
$$
The modulation factor  $(1 - p_{t})^{\gamma}$  in Focal Loss is crucial as it down-weights the loss for well-classified examples and focuses learning on hard examples. However, Focal Loss treats all classes equally with the same modulation factor, which does not address the imbalance among foreground classes.
Adaptive Class-Wise Focusing Factor. To tackle both the foreground-background imbalance and the inter-foreground class imbalance, we introduce an adaptive, class-wise focusing factor  $\gamma^{t}$  that adjusts according to the imbalance degree of each class i. The adaptive focusing factor  $\gamma^{i}$  is defined as:
$$
\begin{array}{l} \gamma^ {i} = \gamma_ {a} + \gamma_ {b} ^ {i} \tag {11} \\ = \gamma_ {a} + s (1 - \eta^ {i}) \\ \end{array}
$$

---

## Page 6

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
6
Here,  $\gamma^{i}$  is decomposed into a class-agnostic parameter  $\gamma_{a}$  and a class-specific parameter  $\gamma_{b}^{i}$ . The parameter  $\gamma_{a}$  represents the basic focusing factor under balanced data scenarios, while  $\gamma_{b}^{i} \geq 0$  is a variable parameter related to the imbalance degree of class i. The term  $\eta^{i} = N_{i}/N$  where N is the total number of points in the point cloud, and  $N_{i}$  is the number of points in class i. The value of  $\eta^{i}$  is constrained to the range [0, 1], and  $1 - \eta^{i}$  inversely reflects the weight for low-frequency classes. The hyperparameter s is a scaling factor that determines the upper limit of  $\gamma^{i}$ .
Dynamic Weight Factor. While the adaptive focusing factor  $\gamma^{i}$  ensures more loss contribution from rare samples, it does not fully resolve the class imbalance problem. Therefore, we introduce a dynamic weighting factor  $\omega^{i}$  to provide higher weights for rare classes:
$$
\omega^ {i} = \frac {1}{\log (\delta + \eta^ {i})} \tag {12}
$$
where  $\delta$  is a small constant to prevent division by zero.
Combining these components, the final ACE Loss is expressed as:
$$
\begin{array}{l} \mathcal {L} _ {A C E} \left(p _ {i}\right) = - \alpha_ {i} \omega^ {i} \left(1 - p _ {i}\right) ^ {\gamma^ {i}} \log \left(p _ {i}\right) \\ = - \sum_ {i = 1} ^ {C} \alpha_ {i} \frac {1}{\log (\delta + \eta^ {i})} (1 - p _ {i}) ^ {\gamma_ {a} + \eta_ {b} ^ {i}} \log (p _ {i}) \tag {13} \\ \end{array}
$$
The ACE Loss effectively prioritizes the learning of rare class samples by dynamically adjusting both the focusing factor and the class weights based on the distribution of point cloud data, thereby addressing the critical issue of class imbalance in curb detection.
2) Lovász-Sofimax Loss
Lovász Loss is particularly effective in handling imbalanced datasets and excels in addressing sparse boundary issues  $[50]$ . Compared to traditional cross-entropy loss, it demonstrates superior performance in terms of Intersection over Union (IoU) scores. For a given true label vector  $y^{*}$  and a predicted label vector  $\tilde{y}$ , the IoU index for class c is defined as:
$$
\mathrm{IoU} _ {c} \left(\boldsymbol {y} ^ {*}, \widetilde {\boldsymbol {y}}\right) = \frac {\left| \left\{\boldsymbol {y} ^ {*} = c \right\} \cap \left\{\widetilde {\boldsymbol {y}} = c \right\} \right|}{\left| \left\{\boldsymbol {y} ^ {*} = c \right\} \cup \left\{\widetilde {\boldsymbol {y}} = c \right\} \right|} \tag {14}
$$
This index provides the ratio between the intersection and union of the true and predicted masks within the range  $[0, 1]$ , with the convention 0/0 = 1. The corresponding loss function employed in empirical risk minimization is:
$$
\Delta_ {\mathrm{IoU} _ {c}} \left(\boldsymbol {y} ^ {*}, \widetilde {\boldsymbol {y}}\right) = 1 - \mathrm{IoU} _ {c} \left(\boldsymbol {y} ^ {*}, \widetilde {\boldsymbol {y}}\right) \tag {15}
$$
For multi-label datasets, it is customary to average across classes, yielding the Mean IoU (mIoU).
The Lovász-Softmax loss extends this concept by applying the Lovász extension to the softmax probabilities of a model's output. It optimizes a convex surrogate of the IoU score, which is more suitable for gradient-based optimization. Specifically, the loss  $L_{IoU}$  for a set of classes C is defined as:
$$
\mathcal {L} _ {I o U} (\boldsymbol {y} ^ {\star}, \widetilde {\boldsymbol {y}}) = \sum_ {c \in C} \Delta_ {\mathrm{IoU} _ {c}} (\boldsymbol {y} ^ {\star}, \widetilde {\boldsymbol {y}}) \tag {16}
$$
![](page_0005/images/0.jpg)

Fig. 5. Process of multiple clustering and fitting to remove noise points. The left figure shows the effect of multiple clustering in discontinuous scenes. The right figure shows the method of curve fitting and setting distance to remove noise points.
The computation involves ordering the pixels by error margin and computing a weighted sum of the individual errors, thus directly targeting the errors that most impact the IoU score.
E. Multi-Cluster and Curve Fitting
This paper introduces a post-processing method based on multi-cluster refitting to filter noise points from LiDAR data segmentation results, thereby enhancing the detection accuracy of curbs. Due to the increasing sparsity of LiDAR point clouds with distance and the potential interruption of curb lines due to obstructions, direct curb clustering along the sides of roads is challenging, as shown in Fig. 5. Thus, we adopt a multi-cluster strategy, treating the curb in multiple segments.
To address this challenge, we initially apply the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm  $[51]$  for preliminary segmentation of the detected curbs. DBSCAN is characterized by its ability to identify clusters of arbitrary shapes without a predefined number of clusters, efficiently handling noise points. The core idea of DBSCAN revolves around setting a neighborhood radius  $\varepsilon$  (Eps) and a minimum sample number minPts (min-samples) to determine cluster membership. In our study, we set Eps to 1 and min-samples to 5.
Let $P$ be a point in the point cloud; its $\varepsilon$-neighborhood, denoted as $N_{\varepsilon}(P)$, is defined as:
$$
N _ {\varepsilon} (P) = \{Q \in \text { Dataset } \mid \operatorname{dist} (P, Q) \leq \varepsilon \} \tag {17}
$$
where  $\text{dist}(P,Q)$  represents the distance between points P and Q. P is considered a core point if its  $\varepsilon$ -neighborhood contains at least minPts points, i.e.,  $|N_{\varepsilon}(P)| \geq \min Pts$ .
Post-clustering, we fit polynomial curves to each independent curb segment. The key is to precisely fit the geometric shape of the curb while eliminating noise points not belonging to the curb. After polynomial curve fitting of each segment, by calculating the distance from points to the fitted curve, we can

---

## Page 7

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
7
effectively identify and eliminate noise points located outside the fitted curve, as shown in Fig. 5.
During the polynomial curve fitting, the aim is to eliminate noise points not belonging to the curb. Assuming the curve equation is  $f(x)$ , for any point  $P(x,y)$  in the point cloud, we calculate the perpendicular distance d to the curve:
$$
d (P, f) = | y - f (x) | \tag {18}
$$
If $d(P, f)$ exceeds a predetermined threshold $\delta$, the point $P$ is considered noise and is removed from the dataset:
$$
\text { If } d (P, f) > \delta , \text { then } P \text { is   noise } \tag {19}
$$
The complete operation process is shown in Algorithm. 1. In order to improve calculation efficiency, we use parallel computing and KDTree's efficient point cloud search scheme. Through this approach, combining the DBSCAN algorithm with polynomial curve fitting effectively identifies and extracts accurate curb lines, while eliminating noise points, thus improving the overall detection accuracy.
Algorithm 1 Multi-Cluster and Curve Fitting Post-Processing Require: Point cloud data, $\varepsilon$, minPts, $\delta$
Ensure: Refined curb line segmentation
1: Step 1: Apply DBSCAN to point cloud data with parallel processing
2: for each point P in point cloud do
3:    Compute  $N_{\varepsilon}(P)$  in parallel computation
4:    if  $|N_{\varepsilon}(P)| \geq \minPts$  then
5:    Mark P as a core point
6:    end if
7: end for
8: Step 2: Segment curb lines into clusters
9: Step 3: Fit polynomial curves in batch processing
10: for each curb segment do
11:    Fit polynomial curve  $f(x)$ 
12:    for each point  $P(x, y)$  in segment do
13:    Calculate perpendicular distance  $d(P, f)$ 
14:    Build KDTree for efficient nearest neighbor search
15:    if  $d(P, f) > \delta$  then
16:    Remove P as noise
17:    end if
18:    end for
19: end for
20: Step 4: Output refined curb line segments
IV. EXPERIMENT AND ANALYSIS
A. Experiment Setup
1) Training Details
Our model training was conducted in an Ubuntu 20.04 environment, utilizing an Intel(R) Xeon(R) Gold 5318S CPU @ 2.10GHz and an NVIDIA RTX 3090 GPU. We employed the PyTorch framework for model training and set training parameters with a batch size of 6, a total of 100 epochs, and a learning rate of 0.001.
Regarding the datasets, we employed two distinct datasets for training and evaluation: the publicly available NRS dataset
TABLE II COMPARISON OF RESULTS IN NRS-DATASET [26]. (W/O) AND (W) REPRESENT WITHOUT OR WITH THE AUXILIARY TRAINING OF ROAD AND SIDEWALK LABELS RESPECTIVELY.
<table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>F-1 score</td></tr><tr><td>PointPillars [52]</td><td>0.759</td><td>0.6019</td><td>0.6524</td></tr><tr><td>U-Net [37]</td><td>0.7546</td><td>0.7018</td><td>0.7172</td></tr><tr><td>Swin-T [53]</td><td>0.7216</td><td>0.7034</td><td>0.7001</td></tr><tr><td>CSWin-T [54]</td><td>0.7692</td><td>0.6597</td><td>0.6966</td></tr><tr><td>LCDeT [26]</td><td>0.8257</td><td>0.8050</td><td>0.8092</td></tr><tr><td>CurbNet (w/o)</td><td>0.8225</td><td>0.8264</td><td>0.8234</td></tr><tr><td>CurbNet (w)</td><td>0.8281</td><td>0.8329</td><td>0.8308</td></tr><tr><td>CurbNet-post</td><td>0.8420</td><td>0.8496</td><td>0.8457</td></tr></table>
and our custom-built 3D-Curb dataset. Both datasets comprised curb data collected from forward-facing trajectories, each extending over 40 meters. To enhance feature learning for curb detection, we augmented the training process by including two additional categories: road and sidewalk.
2) Evaluation Metrics
The performance of our curb detection method was rigorously evaluated using standard metrics. These include Precision, Recall, and F-1 score, which are quintessential for quantifying the accuracy and reliability of classification models. Precision, defined as  $\frac{TP}{TP+FP}$ , measures the proportion of correctly predicted positive observations to the total predicted positives. Recall, calculated as  $\frac{TP}{TP+FN}$ , assesses the proportion of actual positives that were correctly identified. The F-1 score, given by  $2 \times \frac{Precision \times Recall}{Precision + Recall}$ , harmonizes the balance between Precision and Recall, providing a single measure of efficacy. Here, TP (True Positives) represents the number of correct positive predictions, FP (False Positives) denotes the count of negative instances incorrectly classified as positive, TN (True Negatives) refers to the count of correct negative predictions, and FN (False Negatives) signifies the instances where positive cases were wrongly predicted as negative. These metrics collectively offer a comprehensive view of our model's performance, which is crucial for the validation.
B. Quantitative Results of Curb Detection
1) Model Training Results
In the NRS dataset experiments (refer to Table II), this study compared classic segmentation algorithms such as PointPillars, U-Net, Swin-Transformer, and CSWin-Transformer, as well as the state-of-the-art curb detection model, LCDet. Leveraging the specially designed MSCA module for 3D curb scenarios, CurbNet achieved the highest detection performance on the NRS dataset. With the auxiliary training of relevant labels, CurbNet attained Precision, Recall, and F-1 scores of 0.8281, 0.8329, and 0.8308, respectively. Among them, auxiliary training helps improve Precision by 0.5 points, and post-processing helps improve Precision by 1.4 points.
The experiments in the 3D-Curb dataset not only compared classic and advanced deep learning model algorithms but also included three traditional methods of manual feature

---

## Page 8

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
8
TABLE III COMPARISON OF RESULTS IN 3D-CURB DATASET. (W/O) AND (W/) REPRESENT WITHOUT OR WITH THE AUXILIARY TRAINING OF ROAD AND SIDEWALK LABELS RESPECTIVELY.
<table><tr><td>Difference</td><td>Methods</td><td>Precision</td><td>Recall</td><td>F-1 score</td></tr><tr><td rowspan="3">Manual Feature</td><td>Zhang [19]</td><td>0.6854</td><td>0.6564</td><td>0.6053</td></tr><tr><td>Sun [8]</td><td>0.6878</td><td>0.6864</td><td>0.6297</td></tr><tr><td>Wang [14]</td><td>0.7209</td><td>0.7013</td><td>0.6973</td></tr><tr><td rowspan="6">Deep Learning</td><td>3D U-Net [40]</td><td>0.7695</td><td>0.7492</td><td>0.7592</td></tr><tr><td>Cylinder3D [41]</td><td>0.8049</td><td>0.8038</td><td>0.7942</td></tr><tr><td>PVKD [42]</td><td>0.8125</td><td>0.8089</td><td>0.8025</td></tr><tr><td>CurbNet (w/o)</td><td>0.8178</td><td>0.8395</td><td>0.8325</td></tr><tr><td>CurbNet (w/)</td><td>0.8292</td><td>0.8567</td><td>0.8427</td></tr><tr><td>CurbNet-post</td><td>0.8743</td><td>0.8647</td><td>0.8695</td></tr></table>
extraction, as shown in Table III. Owing to the robust automatic feature extraction capabilities, deep learning methods significantly outperformed in curb detection, surpassing the other methods by over 10 points in Precision, Recall, and F-1 score. Among the deep learning methods, CurbNet surpassed the best-performing supervised learning model on the SemanticKITTI dataset, Cylinder3D, by 2.5 points in Precision, and exceeded the knowledge distillation model PVKD by 1.5 points. Among them, Precision is improved by more than 1 point through auxiliary training, and by 4.5 points through post-processing.
2) Tolerance Results
Since curbs resemble elongated curves, relevant research often further tests model performance within a certain error range. Experiments are typically conducted in meters and pixels, with 1 pixel approximately equal to 0.1m of Tolerance, and common experimental settings range from 0.1m to 0.4m.
In our study, we conducted Tolerance performance tests on the 3D-Curb dataset, setting four Tolerances ranging from just 0.05m to 0.2m, as shown in Table IV. Tests were carried out on four models: 3D U-Net, Cylinder3D, PVKD, and CurbNet (ours). With the increase in error Tolerance, performance metrics improved significantly. At 0.05m Tolerance, Precision improved by an average of 4 points; at 0.1m, by 9 points; at 0.15m, by 12 points; and at 0.2m, by 13 points. As Tolerance increased, performance gains gradually reached saturation. Notably, our CurbNet exceeded 0.95 in the average values of Precision, Recall, and F-1 score at just 0.15m Tolerance. This represents the optimal performance achieved in curb detection based on point cloud segmentation.
C. Visualization Results of Curb Detection
We conducted a visual analysis of the test results obtained using our CurbNet model, comparing them with the ground truth to further validate the model's detection performance. Fig. 6 illustrates the visualization results in scenarios without obstructions, showcasing the curb detection results in five common road scenes: straight road, curved road, right-angle intersection, curved intersection, and cross intersection. The images clearly demonstrate that the CurbNet model successfully identified all curb lines present in the ground truth, even
in complex intersection scenarios. Notably, in the curved intersection scenario, the results identified by CurbNet were even more precise than the ground truth annotations, showcasing our model's exceptional feature extraction capabilities and scene generalizability.
Fig. 7 also displays the visual recognition results in scenarios with obstructions, where yellow dashed circles highlight the obstructed areas. Despite the absence of point clouds in obstructed areas, the model accurately identified curbs in the other parts without being influenced by the obstructed regions. As long as the input data contained curb features, CurbNet could accurately detect them, unaffected by the obscured areas.
Furthermore, we conducted a visual analysis of the results on the NRS dataset, as shown in Fig. 8. This analysis primarily showcases curb detection in five common road scenarios and three special intersection types. Through a comparative visualization of the detection results and ground truth, our method accurately identified the respective curb features. The NRS dataset includes several challenging scenes characterized by irregular road structures and unique curb configurations, such as sharp elevation changes, occlusions, and narrow pathways. As shown in Fig. 8 (f)-(h), our method successfully detected the respective curb features in these biased scenarios. This demonstrates the robustness of CurbNet in handling dataset bias and its ability to generalize effectively to complex and irregular road conditions. The consistent detection results in these scenarios validate the model's capability to adapt to diverse real-world environments, highlighting its potential for broader applicability in intelligent driving systems.
D. Post-Processing Experiment
In this paper, we conducted controlled experiments to compare the proposed post-processing method. Given the use of multiple clustering followed by curve fitting in post-processing, the setting of clustering parameters plays a crucial role in its effectiveness. Based on the road width and point cloud density characteristics of the 3D-Curb dataset, we experimented with varying the distance variable Eps (from 1m to 4m) and the minimum sample points variable minPts (from 2 to 200), as illustrated in Fig. 9.
As the clustering distance Eps increases, changes in the performance metrics of post-processing become more gradual and similar. However, when the minimum sample points minPts are lower, the performance decreases compared to when Eps is 1m. This is attributed to the larger curb clustering caused by greater clustering distances, resulting in minimal changes post-curve fitting. Additionally, larger clusters tend to overlook sparsely distributed point clouds during curve fitting, thereby reducing performance.
Fig. 9 clearly demonstrates that as the minimum sample points variable minPts increases, the Precision metric of post-processing gradually improves, but both Recall and F-1 score metrics significantly decrease, especially at Eps settings of 1m and 2m. This decline is due to the increase in the number of minPts leading to the neglect of sparsely distributed curb point clouds at greater distances, resulting in a noticeable drop in Recall and F-1 scores.

---

## Page 9

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
9
TABLE IV COMPARISON OF DIFFERENT TOLERANCES IN 3D-CURB DATASET. (W/O) AND (W/) REPRESENT WITHOUT OR WITH THE AUXILIARY TRAINING OF ROAD AND SIDEWALK LABELS RESPECTIVELY.
<table><tr><td rowspan="2">Tolerance (m)</td><td colspan="4">Precision</td><td colspan="4">Recall</td><td colspan="4">F-1 score</td></tr><tr><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.05</td><td>0.10</td><td>0.150.20</td></tr><tr><td>3D U-Net [40]</td><td>0.8293</td><td>0.8583</td><td>0.8881</td><td>0.9012</td><td>0.8078</td><td>0.8498</td><td>0.8745</td><td>0.8843</td><td>0.7933</td><td>0.8389</td><td>0.8662 0.8777</td></tr><tr><td>Cylinder3D [41]</td><td>0.8403</td><td>0.8909</td><td>0.9221</td><td>0.9360</td><td>0.8588</td><td>0.9007</td><td>0.9253</td><td>0.9355</td><td>0.8391</td><td>0.8856</td><td>0.9136 0.9257</td></tr><tr><td>PVKD [42]</td><td>0.8422</td><td>0.8898</td><td>0.9264</td><td>0.9398</td><td>0.8595</td><td>0.9021</td><td>0.9305</td><td>0.9417</td><td>0.8419</td><td>0.8902</td><td>0.9188 0.9343</td></tr><tr><td>CurbNet (w/o)</td><td>0.8588</td><td>0.9091</td><td>0.9397</td><td>0.9503</td><td>0.8934</td><td>0.9355</td><td>0.9559</td><td>0.9689</td><td>0.8793</td><td>0.9211</td><td>0.9476 0.9597</td></tr><tr><td>CurbNet (w/)</td><td>0.8667</td><td>0.9158</td><td>0.9461</td><td>0.9587</td><td>0.9029</td><td>0.9434</td><td>0.9663</td><td>0.9760</td><td>0.8845</td><td>0.9295</td><td>0.9562 0.9673</td></tr><tr><td>CurbNet-post</td><td>0.9139</td><td>0.9530</td><td>0.9679</td><td>0.9738</td><td>0.8976</td><td>0.9303</td><td>0.9441</td><td>0.9501</td><td>0.9056</td><td>0.9413</td><td>0.9558 0.9617</td></tr></table>
![](page_0008/images/0.jpg)

Fig. 6. Curb detection result in 3D-Curb dataset. We compared the curb detection results at five classic intersections. The model accurately detected the curb area and was even better than the ground truth on the curve road.
![](page_0008/images/17.jpg)

Fig. 7. Curb detection result in 3D-Curb dataset under occlusion. We compared the curb detection results at five classic intersections with occlusion. Even under occlusion, it does not affect curb detection in other areas.

---

## Page 10

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
10
![](page_0009/images/0.jpg)

Fig. 8. Curb detection result in NRS Dataset. The excellent detection performance and generalization of the proposed method are well demonstrated on the NRS dataset, and accurate detection can be performed even at the special intersection (f-h).
![](page_0009/images/19.jpg)

![](page_0009/images/20.jpg)

![](page_0009/images/21.jpg)

![](page_0009/images/22.jpg)

Fig. 9. Parameter adjustment experiments for multiple clustering and fitting. We conduct control variable experiments on the main parameter distance variable Eps and the minimum sample point variable minPts of DBSCAN clustering.
Based on comparative experimental results and trade-off between metrics, smaller Eps distances and fewer minPts numbers yield the most optimal post-processing outcomes.
E. Ablation Study
As shown in Table V, this study conducted comparative ablation experiments focusing on the main loss functions and crucial module designs of our model. We first conducted individual experiments on the employed  $L_{CE}$  (Cross-Entropy Loss),  $L_{FL}$  (Focal Loss),  $L_{ACE}$  (Adaptive Cross-Entropy Loss), and  $L_{IoU}$  (Intersection over Union Loss). Subsequently, we tested combinations of  $L_{CE} + L_{IoU}$  loss,  $L_{FL} + L_{IoU}$  loss and  $L_{ACE} + L_{IoU}$  loss. In the experimental results, due to the design of the  $L_{ACE}$  loss addressing the imbalance in the number of curb point clouds compared to other categories, its disproportionate weight settings caused the model to overly focus on recall during training. However, the interaction with
the  $L_{IoU}$  loss led to a more balanced overall performance, achieving optimal detection capabilities. The combination of  $L_{ACE}+L_{IoU}$  loss outperformed the  $L_{CE}+L_{IoU}$  loss group by 1.3 points in Precision and 2.2 points in Recall. It also surpassed the  $L_{FL}+L_{IoU}$  loss group with improvements of 0.7 points in Precision and 2 points in Recall.
Finally, we compared the model's performance with and without the MSCA module. When the MSCA module was not utilized, the model relied on the original encoder-decoder structure from Cylinder3D [41]. Under the $\mathcal{L}_{ACE} + \mathcal{L}_{IoU}$ loss setting, the inclusion of the MSCA module significantly enhanced the performance metrics, particularly increasing Recall by 2 points and the F-1 score by 1 point, which further demonstrates the MSCA module's effectiveness in improving curb detection performance.

---

## Page 11

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
11
TABLE V ABLATION OF DIFFERENT LOSS FUNCTIONS AND MODULES
<table><tr><td>$\mathcal{L}_{CE}$</td><td>$\mathcal{L}_{FL}$</td><td>$\mathcal{L}_{ACE}$</td><td>$\mathcal{L}_{IoU}$</td><td>MSCA</td><td>Precision</td><td>Recall</td><td>F-1</td></tr><tr><td>√</td><td></td><td></td><td></td><td>√</td><td>0.8174</td><td>0.8303</td><td>0.8238</td></tr><tr><td></td><td>√</td><td></td><td></td><td>√</td><td>0.8196</td><td>0.8346</td><td>0.8273</td></tr><tr><td></td><td></td><td>√</td><td></td><td>√</td><td>0.7933</td><td>0.8707</td><td>0.8355</td></tr><tr><td></td><td></td><td></td><td>√</td><td>√</td><td>0.8234</td><td>0.8367</td><td>0.8339</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>√</td><td>0.8186</td><td>0.8472</td><td>0.8374</td></tr><tr><td></td><td>√</td><td></td><td>√</td><td>√</td><td>0.8241</td><td>0.8498</td><td>0.8401</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td></td><td>0.8297</td><td>0.8496</td><td>0.8395</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>√</td><td>0.8311</td><td>0.8695</td><td>0.8499</td></tr></table>
![](page_0010/images/0.jpg)

Fig. 10. Setup of real scene experiment. We used the delivery vehicle equipped with LiDAR to conduct real scene experiments. There are a total of five experimental sections distributed on the HKUST Guangzhou campus.
F. Real Scene Experiment
To further validate the performance and effectiveness of the proposed method, we conducted real-world experiments in addition to the dataset experiments. As depicted in Fig. 10, we utilized an autonomous delivery vehicle as our experimental platform, equipping it with a LiDAR sensor system mounted on its top. The LiDAR used was the OS1-U model with 128 lines, manufactured by Ouster. To ensure the generalizability of our experiments, the autonomous vehicle was driven on roads within the HKUST Guangzhou campus. Data collection and curb detection were carried out in five different road segments (Scene A, B, C, D, and E), as shown in the map in Fig. 10. These segments included both standard road scenarios (straight roads, bends, and intersections) and complex ones (roundabout turns, and intersections) junctions.
The curb detection results in the 5 scenes are illustrated in Fig. 11, where we selected 8 representative images from each scene for visual analysis. Overall, our method achieved commendable curb detection results in each scene, further demonstrating the robust curb feature extraction capability of the CurbNet model. In individual cases, our proposed method accurately detected not only the evident, extensive curbs, but also achieved remarkable results on small-scale curbs, which are typically less conspicuous and easily overlooked, as seen in Scene B (image d) and Scene D (images b and c). This success can be attributed to the model's design focusing on multi-scale feature fusion and height channel feature extraction. Similarly,
TABLE VI
RESULTS IN FIVE REAL SCENES.
<table><tr><td>Site</td><td>Precision</td><td>Recall</td><td>F-1 score</td></tr><tr><td>Scene A</td><td>0.8331</td><td>0.8821</td><td>0.8569</td></tr><tr><td>Scene B</td><td>0.8950</td><td>0.8123</td><td>0.8517</td></tr><tr><td>Scene C</td><td>0.8610</td><td>0.8366</td><td>0.8530</td></tr><tr><td>Scene D</td><td>0.8018</td><td>0.8533</td><td>0.8268</td></tr><tr><td>Scene E</td><td>0.8579</td><td>0.8388</td><td>0.8483</td></tr><tr><td>All Scene</td><td>0.8462</td><td>0.8443</td><td>0.8453</td></tr></table>
our method also exhibited superior detection performance in complex intersection scenarios with irregular curb distributions (Scene B and Scene C). Particularly in Scene B, the method precisely detected all curbs present in the LiDAR point cloud.
Finally, we also quantitatively evaluate the real scene experiments by testing key metrics, as shown in Table VI. By comparing with manually annotated ground truth, Curb-Net achieved an average Precision, Recall, and F1 score of 0.8462, 0.8443, and 0.8453, respectively, across the five scenes. Among them, the Recall and F-1 score indicators obtained in Scene A are the highest, which are 0.8821 and 0.8569 respectively, and the Precision indicator obtained in Scene B is the highest 0.8950. These results corroborate with those obtained from dataset testing, further substantiating the excellent performance and generalizability of the CurbNet.
G. Time Consumption
In practical applications for intelligent vehicles, real-time curb detection is critical. As shown in Table VII, we evaluated the time consumption for both the model inference and post-processing stages of our curb detection framework. To ensure the efficiency of the entire processing framework, the model inference is executed on the GPU, while the post-processing runs on the CPU.
In various scenarios, since the LiDAR used and the number of input point clouds are different, the time consumption of computational processing is also different. Nevertheless, the CurbNet framework consistently achieves an overall real-time performance exceeding 15 Hz during both model inference and post-processing stages. Notably, the post-processing stage operates faster than the model inference stage, ensuring the smooth and efficient operation of the entire framework.
Furthermore, we evaluated the real-time performance of CurbNet on an on-board processing unit, specifically the NVIDIA Jetson AGX Orin [55], which provides up to 275 TOPS of computational power with INT8 precision. Through model optimization using TensorRT and INT8 precision inference, the CurbNet achieved a processing speed exceeding 20 FPS on this device. These results underscore the feasibility of deploying CurbNet on modern edge computing platforms, ensuring real-time curb detection even under resource-constrained conditions. This highlights the practicality of CurbNet for real-world autonomous driving applications.

---

## Page 12

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
12
![](page_0011/images/0.jpg)

Fig. 11. Curb detection results of real experiments in five field scenarios. Our method achieves excellent curb detection results in five real-world scenarios. In particular, it also shows excellent performance in complex intersection scenarios B and C with irregular curb distribution.
TABLE VII COMPARISON OF TIME CONSUMPTION IN MODEL INFERENCE AND POST-PROCESSING, THE UNIT IS (FPS/MS).
<table><tr><td>Stage</td><td>3D-Curb</td><td>NRS</td><td>Real Scene</td></tr><tr><td>Model Inference</td><td>20.69 / 48.33</td><td>18.18 / 55.01</td><td>16.22 / 61.65</td></tr><tr><td>Post-Processing</td><td>26.96 / 37.09</td><td>26.08 / 38.34</td><td>19.02 / 52.58</td></tr></table>
V. LIMITATION
While the method presented in this paper demonstrates effective and accurate detection of curbs in road scenes, thus providing a basis for navigable area determination for autonomous driving, it currently has limitations in detecting curbs solely within the LiDAR point cloud. Due to factors such as the scanning angle, field of view, and obstructions inherent to LiDAR technology, some road areas remain undetected in the point cloud, leading to an inability of the model to extract corresponding curb features. This necessitates future research involving more advanced sensors to minimize scanning blind spots. Additionally, the development of a model incorporating a curb prediction module that operates effectively in areas without scanning blind spots is essential to mitigate the impact of these blind spots on curb detection.
VI. CONCLUSION
In this paper, we established the 3D-Curb dataset, comprising 7,100 frames. To our knowledge, this is currently the largest and most diverse curb point cloud dataset with the
most extensive range of annotated categories. Notably, this is also the first dataset to feature 3D point cloud annotations for curbs, which will significantly aid future related research. Within the CurbNet framework, we introduced the Multi-Scale and Channel Attention (MSCA) module, addressing the challenges of uneven distribution of curb features and the reliance on high-frequency z-axis features. Additionally, we introduce a novel adaptive loss function group to resolve the imbalance in the number of curb point clouds relative to other categories. Extensive experiments on both the NRS and 3D-Curb datasets demonstrated that our approach outperforms the current leading curb detection and point cloud segmentation models. In the tolerance experiments, CurbNet achieved over 0.95 average performance in Precision, Recall, and F-1 score metrics at just 0.15m tolerance, setting a new standard. Furthermore, our post-processing approach of multi-clustering and curve fitting effectively eliminated noise in the curb results, enhancing the Precision, Recall, and F-1 score metrics to 0.8744, 0.8648, and 0.8696, respectively. Finally, the excellent detection performance and generalization of our proposed method were further verified in real scene experiments.
The 3D-Curb dataset and the CurbNet framework established in this study lay a foundation for future research in curb detection. In our upcoming research, we plan to create a more comprehensive dataset incorporating additional modalities. Similarly, we aim to explore and enhance the capabilities of the CurbNet framework, improving its performance in multimodal data contexts.

---

## Page 13

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
13
REFERENCES
[1] T. Luettel, M. Himmelsbach, and H.-J. Wuensche, “Autonomous ground vehicles—concepts and a path to the future,” Proceedings of the IEEE, vol. 100, no. Special Centennial Issue, pp. 1831–1839, 2012.
[2] F. Ma, X. Yan, Y. Liu, and M. Liu, “Every dataset counts: Scaling up monocular 3D object detection with joint datasets training,” arXiv preprint arXiv:2310.00920, 2023.
[3] Z. Xu, Y. Sun, and M. Liu, “Icub: Imitation learning-based detection of road curbs using aerial images for autonomous driving,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 1097–1104, 2021.
[4] J. K. Suhr, J. Jang, D. Min, and H. G. Jung, “Sensor fusion-based low-cost vehicle localization system for complex urban environments,” IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 5, pp. 1078–1086, 2016.
[5] A. Y. Hata, F. T. Ramos, and D. F. Wolf, “Monte carlo localization on gaussian process occupancy maps for urban environments,” IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 9, pp. 2893–2902, 2017.
[6] L. M. Romero, J. A. Guerrero, and G. Romero, “Road curb detection: A historical survey,” Sensors, vol. 21, no. 21, p. 6952, 2021.
[7] A. B. Hillel, R. Lerner, D. Levi, and G. Raz, “Recent progress in road and lane detection: a survey,” Machine Vision and Applications, vol. 25, no. 3, pp. 727–745, 2014.
[8] P. Sun, X. Zhao, Z. Xu, R. Wang, and H. Min, “A 3D lidar data-based dedicated road boundary detection algorithm for autonomous vehicles,” IEEE Access, vol. 7, pp. 29,623–29,638, 2019.
[9] C. Wei, H. Li, J. Shi, G. Zhao, H. Feng, and L. Quan, “Row anchor selection: classification method for early-stage crop row-following,” Computers and Electronics in Agriculture, vol. 192, p. 106577, 2022.
[10] G. Zhao, Y. Liu, W. Qi, F. Ma, M. Liu, and J. Ma, “Fisheyedepth: A real scale self-supervised depth estimation model for fisheye camera,” arXiv preprint arXiv:2409.15054, 2024.
[11] F. Ma, S. Wang, and M. Liu, “An automatic multi-fidlar extrinsic calibration algorithm using corner planes,” in 2022 IEEE International Conference on Robotics and Biomimetics, 2022, pp. 235–240.
[12] S. O. Demir, T. E. Ertop, A. B. Koku, and E. I. Konukseven, “An adaptive approach for road boundary detection using 2d lidar sensor,” in 2017 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems, 2017, pp. 206–211.
[13] E. Horváth, C. Pozna, and M. Unger, “Real-time lidar-based urban road and sidewalk detection for autonomous vehicles,” Sensors, 2021.
[14] G. Wang, J. Wu, R. He, and B. Tian, “Speed and accuracy tradeoff for lidar data based road boundary detection,” IEEE/CAA Journal of Automatica Sinica, vol. 8, no. 6, pp. 1210–1220, 2020.
[15] T. Chen, B. Dai, D. Liu, J. Song, and Z. Liu, “Velodyne-based curb detection up to 50 meters away,” in 2015 IEEE Intelligent Vehicles Symposium, 2015, pp. 241–248.
[16] A. Y. Hata, F. S. Osorio, and D. F. Wolf, “Robust curb detection and vehicle localization in urban environments,” in 2014 IEEE Intelligent Vehicles Symposium Proceedings, 2014, pp. 1257–1261.
[17] L. Zhou and G. Vosselman, “Mapping curbshores in airborne and mobile laser scanning data,” International Journal of Applied Earth Observation and Geoinformation, vol. 18, pp. 293–304, 2012.
[18] S. Xu, R. Wang, and H. Zheng, “Road curb extraction from mobile lidar point clouds,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 2, pp. 996–1009, 2016.
[19] Y. Zhang, J. Wang, X. Wang, and J. M. Dolan, “Road-segmentation-based curb detection method for self-driving via a 3D-LiDAR sensor,” IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 12, pp. 3981–3991, 2018.
[20] B. Qin, Z. Chong, T. Bandyopadhyay, M. H. Ang, E. Frazzoli, and D. Rus, “Curb-intersection feature based monte carlo localization on urban roads,” in 2012 IEEE International Conference on Robotics and Automation, 2012, pp. 2640–2646.
[21] W. Qi, G. Zhao, F. Ma, L. Zheng, and M. Liu, “Clrkdnet: Speeding up lane detection with knowledge distillation,” arXiv preprint arXiv:2405.12503, 2024.
[22] G. Zhao, L. Quan, H. Li, H. Feng, S. Li, S. Zhang, and R. Liu, “Real-time recognition system of soybean seed full-surface defects based on deep learning,” Computers and Electronics in Agriculture, vol. 187, p. 106230, 2021.
[23] Z. Xu, Y. Sun, L. Wang, and M. Liu, “Cp-loss: Connectivity-preserving loss for road curb detection in autonomous driving with aerial images,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2021, pp. 1117–1123.
[24] Y. Jung, M. Jeon, C. Kim, S.-W. Seo, and S.-W. Kim, “Uncertainty-aware fast curb detection using convolutional networks in point clouds,” in 2021 IEEE International Conference on Robotics and Automation, 2021, pp. 12882–12888.
[25] T. Suleymanov, L. Kunze, and P. Newman, “Online inference and detection of curbs in partially occluded scenes with sparse lidar,” in 2019 IEEE Intelligent Transportation Systems Conference, 2019, pp. 2693–2700.
[26] J. Gao, H. Jie, B. Xu, L. Liu, J. Hu, and W. Liu, “Lcdet: Lidar curb detection network with transformer,” in 2023 International Joint Conference on Neural Networks, 2023, pp. 1–9.
[27] J. L. Apellániz, M. García, N. Araniuelo, J. Barandián, and M. Nieto, "Lidar-based curb detection for ground truth annotation in automated driving validation," arXiv preprint arXiv:2212.00544, 2023.
[28] D. Zai, J. Li, Y. Guo, M. Cheng, Y. Lin, H. Luo, and C. Wang, “3-d road boundary extraction from mobile laser scanning data via supervoxels and graph cuts,” IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 3, pp. 802–813, 2017.
[29] U. Ozertem and D. Erdogmus, “Locally defined principal curves and surfaces,” The Journal of Machine Learning Research, 2011.
[30] H. Jie, J. Guo, Q. Zhao, Z. Ning, J. Hu, L. Liu, and W. Liu, “An efficient curb detection and tracking method for intelligent vehicles via a high-resolution 3D-LiDAR,” in 4th International Conference on Information Science, Electrical, and Automation Engineering, vol. 12257, 2022, pp. 310–317.
[31] W. Yao, Z. Deng, and L. Zhou, “Road curb detection using 3D lidar and integral laser points for intelligent vehicles,” in The 6th International Conference on Soft Computing and Intelligent Systems, and The 13th International Symposium on Advanced Intelligence Systems, 2012, pp. 100–105.
[32] J. Zhang, S. Singh et al., “Loam: Lidar odometry and mapping in real-time,” in Robotics: Science and systems, vol. 2, no. 9, 2014, pp. 1–9.
[33] A. Y. Hata and D. F. Wolf, “Feature detection for vehicle localization in urban environments using a multilayer lidar,” IEEE Transactions on Intelligent Transportation Systems, vol. 17, no. 2, pp. 420–429, 2015.
[34] B. Yang, L. Fang, and J. Li, “Semi-automated extraction and delineation of 3D roads of street scene from mobile laser scanning point clouds,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 79, pp. 80–93, 2013.
[35] D. Bai, T. Cao, J. Guo, and B. Liu, “How to build a curb dataset with lidar data for autonomous driving,” in 2022 International Conference on Robotics and Automation, 2022, pp. 2576–2582.
[36] J. Liang, N. Homayounfar, W.-C. Ma, S. Wang, and R. Urtasun, "Convolutional recurrent network for road boundary extraction," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 9512-9521.
[37] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 10th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, 2015, pp. 234–241.
[38] X. Pan, J. Shi, P. Luo, X. Wang, and X. Tang, “Spatial as deep: Spatial cnn for traffic scene understanding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.
[39] D. Kukolj, I. Marinović, and S. Menet, “Road edge detection based on combined deep learning and spatial statistics of lidar data,” Journal of Spatial Science Research, vol. 68, no. 2, pp. 245–259, 2023.
[40] O. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D u-net: learning dense volumetric segmentation from sparse annotation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19, 2016, pp. 424–432.
[41] H. Zhou, X. Zhu, X. Song, Y. Ma, Z. Wang, H. Li, and D. Lin, “Cylinder3D: An effective 3D framework for driving-scene lidar semantic segmentation,” arXiv preprint arXiv:2008.01550, 2020.
[42] Y. Hou, X. Zhu, Y. Ma, C. C. Loy, and Y. Li, “Point-to-voxel knowledge distillation for lidar semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8479–8488.
[43] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11621–11631.

---

## Page 14

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
14
[44] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.
[45] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall, “Semantickit: A dataset for semantic scene understanding of lidar sequences,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9297–9307.
[46] D. Zermas, I. Izaz, and N. Papanikolopoulos, “Fast segmentation of 3d point clouds: A paradigm on lidar data for autonomous vehicle applications,” in 2017 IEEE International Conference on Robotics and Automation, 2017, pp. 5067–5073.
[47] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao, “Shape robust text detection with progressive scale expansion network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 9336–9345.
[48] Z. Zhang and M. Sabuncu, “Generalized cross entropy loss for training deep neural networks with noisy labels,” Advances in Neural Information Processing Systems, vol. 31, 2018.
[49] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2980–2988.
[50] S. Jadon, “A survey of loss functions for semantic segmentation,” in 2020 IEEE conference on Computational Intelligence in Bioinformatics and Computational Biology, 2020, pp. 1–7.
[51] E. Schubert, J. Sander, M. Ester, H. P. Kriegel, and X. Xu, “Dbscan revisited, revisited: why and how you should (still) use dbscan,” ACM Transactions on Database Systems, 2017.
[52] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, "Pointpillars: Fast encoders for object detection from point clouds," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 12697-12705.
[53] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10012–10022.
[54] X. Dong, J. Bao, D. Chen, W. Zhang, N. Yu, L. Yuan, D. Chen, and B. Guo, “Cswin transformer: A general vision transformer backbone with cross-shaped windows,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 124–12 134.
[55] NVIDIA Corporation, “NVIDIA Jetson AGX Orin: Ai at the edge,” https://developer.nvidia.com/embedded/jetson-agx-orin, 2022, accessed: January 11, 2025.
![](page_0013/images/0.jpg)

Guoyang Zhao (Student Member, IEEE) received the B.Eng. degree in logistics engineering from Northeast Agricultural University, Harbin, China, in 2022, and the M.Phil. degree in robotics and autonomous systems from The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China, in 2024. He is currently pursuing the Ph.D. degree at the Intelligent Autonomous Driving Center, Robotics and Autonomous Systems Thrust, The Hong Kong University of Science and Technology, Guangzhou, China. His research inter-
ests include computer vision, robotics navigation, and deep learning.
![](page_0013/images/1.jpg)

Fulong Ma received the B.Eng. degree in automation from the University of Science and Technology of China, Hefei, China, in 2018. He is currently pursuing the Ph.D degree with the Robotics and Autonomous Systems Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. His research interests include computer vision, sensor calibration, and deep learning.
![](page_0013/images/2.jpg)

Weiqing Qi received the B.S. degree in Computer Science from University of California, Santa Barbara, CA, USA, in 2021, and the M.Phil. degree in robotics and autonomous systems from The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China, in 2024. His current research interests include lane detection, drivable area segmentation, and semantics segmentation, etc.
![](page_0013/images/3.jpg)

Yuxuan Liu received the B.Eng. degree in Mechatronic from Zhejiang University, Zhejiang, China in 2019, and the Ph.D. degree in Electronic and Computer Engineering, The Hong Kong University of Science and Technology, Hong Kong, China, in 2023. His current research interests include autonomous driving, deep learning, robotics, visual 3D object detection, visual depth prediction, etc.
![](page_0013/images/4.jpg)

Ming Liu received the B.A. degree in automation from Tongji University, Shanghai, China, in 2005, and the Ph.D. degree from the Department of Mechanical and Process Engineering, ETH Zurich, Zurich, Switzerland, in 2013, supervised by Prof. Roland Siegwart. During his master's stay with Tongji University, he stayed one year with the Erlangen-Nunberg University and Fraunhofer Institute ISB, Erlangen, Germany, as a Visiting Scholar. He is currently an Associate Professor with the
Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. He is also a founding member of Shanghai Swing Automation Ltd., Co. He is currently the Chairman of Shenzhen Unity Drive Inc., China. He has coordinated and been involved in NSF Projects and National 863-Hi-TechPlan Projects in China. From 2014 to 2015, He was an Assistant Professor with City University of Hong Kong, Hong Kong SAR, China. He was an Assistant Professor from 2017 to 2020 and an Associate Professor since 2020, with The Hong Kong University of Science and Technology, Hong Kong SAR, China.
He has published several papers in top journals including IEEE Transactions on Robotics and International Journal of Robotics Research. He was an Associate Editor for IEEE Robotics and Automation Letters, IET CyberSystems and Robotics, International Journal of Robotics and Automation, IEEE IROS Conference 2018, 2019 and 2020. He served as a Guest Editor of special issues in IEEE Transactions on Automation Science and Engineering. His research interests include dynamic environment modeling, deep learning for robotics, 3-D mapping, machine learning, and visual control.
![](page_0013/images/5.jpg)

Jun Ma (Senior Member, IEEE) received the B.Eng. degree with First Class Honours in electrical and electronic engineering from Nanyang Technological University, Singapore, in 2014, and the Ph.D. degree in electrical and computer engineering from the National University of Singapore, Singapore, in 2018. From 2018 to 2021, he held several positions at the National University of Singapore; University College London, London, U.K.; University of California, Berkeley, Berkeley, CA, USA; and Harvard University; Cambridge, MA, USA. He is currently an
Assistant Professor with the Department of Autonomous Systems Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China, and also with the Division of Emerging Interdisciplinary Areas, The Hong Kong University of Science and Technology, Hong Kong SAR, China. He is also the Director of Intelligent Autonomous Driving Center, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. His research interests include motion planning and control for robotics and autonomous driving.

---

