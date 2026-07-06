IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
5
volutions (denoted as SConv) with stride values \( s \in \{1,3,5\} \):
\[
\mathbf {X} _ {s 1} = \operatorname{SConv} _ {s = 1} (\mathbf {X}),
\]
\[
\mathbf {X} _ {s 3} = \operatorname{SConv} _ {s = 3} (\mathbf {X}), \tag {1}
\]
\[
\mathbf {X} _ {s 5} = \operatorname{SConv} _ {s = 5} (\mathbf {X})
\]
These convolutions produce feature maps  \( X_{s1} \) ,  \( X_{s3} \) , and  \( X_{s5} \)  at different spatial scales. The multi-scale outputs are fused as follows:
\[
\mathbf {X} _ {\mathrm{ms}} = f (\mathbf {X} _ {s 1}, \mathbf {X} _ {s 3}, \mathbf {X} _ {s 5}) \tag {2}
\]
where  \( f(\cdot) \)  represents the pyramid fusion operation, typically achieved by concatenating the outputs followed by additional sparse convolutions for deep integration.
Channel Attention. The primary distinguishing feature of curbs is the subtle height difference between the road and the sidewalk, reflected in the z-axis of point cloud data. Unlike general point cloud segmentation algorithms such as Cylinder3D [41], which primarily focus on feature learning in the xy plane, we designed the channel attention module to capture high-frequency features along the z-axis. Initially, a  \( 1 \times 1 \times D \)  sparse convolution is applied to preliminarily extract channel features:
\[
\mathbf {C} = \operatorname{SConv} _ {1 \times 1 \times D} (\mathbf {X}) \tag {3}
\]
The channel features are then processed through an encoder-decoder structured MLP to further refine the height feature:
\[
\mathbf {C} _ {\text { encoded }} = \mathrm{MLP} _ {\text { enc }} (\mathbf {C}), \quad \mathbf {C} _ {\text { decoded }} = \mathrm{MLP} _ {\text { dec }} (\mathbf {C} _ {\text { encoded }}) \tag {4}
\]
The output of the MLP,  \( C_{ decoded} \) , is then passed through two parallel branches. In the first branch, a softmax function generates dynamic weights for each channel:
\[
\mathbf {W} _ {\text { channel }} = \operatorname{softmax} (\mathbf {C} _ {\text { decoded }}) \tag {5}
\]
In the second branch,  \( C_{ decoded} \)  is further processed by another  \( 1 \times 1 \times D \)  sparse convolution:
\[
\mathbf {C} _ {\text { conv }} = \mathrm{SCConv} _ {1 \times 1 \times D} (\mathbf {C} _ {\text { decoded }}) \tag {6}
\]
The \(\mathbf{W}_{\mathrm{channel}}\) are element-wise multiplied with the output \(\mathbf{C}_{\mathrm{conv}}\) to produce the final channel-attended feature map:
\[
\mathbf {X} _ {\text { channel }} = \mathbf {W} _ {\text { channel }} \odot \mathbf {C} _ {\text { conv }} \tag {7}
\]
The final output of the MSCA is obtained by combining the multi-scale fusion output and the channel attention output:
\[
\mathbf {X} _ {\text { output }} = \mathbf {X} _ {\mathrm{ms}} + \mathbf {X} _ {\text { channel }} \tag {8}
\]
By combining multi-scale feature extraction with dynamic channel attention, the MSCA module effectively captures both spatial and height variations in point cloud data, thereby enhancing the model's ability to detect curbs.
![](images/0.jpg)

Fig. 4. Structure of multi-scale and channel attention (MSCA) module. SConv means Sparse convolution layer. Multi-Scale Fusion is mainly used to fuse spatial features of different scales, and Channel Attention is used to dynamically extract height features of the z-axis.
D. Loss Group
In real-world scenarios, the point cloud data for curbs comprises only a small fraction compared to other categories such as roads and buildings. Using uniform loss weights can lead to training imbalances, adversely affecting the recognition performance for the minority class, i.e., curbs. To address this, we propose a novel combination of Adaptive Cross-Entropy (ACE) Loss and Lovász-Softmax Loss.
I) Adaptive Cross Entropy (ACE) Loss
Due to the imbalance between the number of curb point clouds and the number of other categories such as roads and buildings. Standard loss functions like Cross Entropy (CE) [48] do not adequately address this imbalance, leading to suboptimal performance in recognizing minority classes. The standard CE loss is defined as:
\[
\mathcal {L} _ {C E} \left(p _ {\mathrm{t}}\right) = - \log \left(p _ {\mathrm{t}}\right) \tag {9}
\]
where \( p_t \) represents the predicted probability of the true class.
Given the disproportionate representation of classes in the point cloud data, we draw inspiration from the Focal Loss [49] to reallocate the loss contribution of easy and hard samples, significantly reducing the influence of the majority background samples:
\[
\mathcal {L} _ {F L} \left(p _ {\mathrm{t}}\right) = - \alpha_ {\mathrm{t}} \left(1 - p _ {\mathrm{t}}\right) ^ {\gamma} \log \left(p _ {\mathrm{t}}\right) \tag {10}
\]
The modulation factor  \( (1 - p_{t})^{\gamma} \)  in Focal Loss is crucial as it down-weights the loss for well-classified examples and focuses learning on hard examples. However, Focal Loss treats all classes equally with the same modulation factor, which does not address the imbalance among foreground classes.
Adaptive Class-Wise Focusing Factor. To tackle both the foreground-background imbalance and the inter-foreground class imbalance, we introduce an adaptive, class-wise focusing factor  \( \gamma^{t} \)  that adjusts according to the imbalance degree of each class i. The adaptive focusing factor  \( \gamma^{i} \)  is defined as:
\[
\begin{array}{l} \gamma^ {i} = \gamma_ {a} + \gamma_ {b} ^ {i} \tag {11} \\ = \gamma_ {a} + s (1 - \eta^ {i}) \\ \end{array}
\]