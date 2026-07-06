IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
7
effectively identify and eliminate noise points located outside the fitted curve, as shown in Fig. 5.
During the polynomial curve fitting, the aim is to eliminate noise points not belonging to the curb. Assuming the curve equation is  \( f(x) \) , for any point  \( P(x,y) \)  in the point cloud, we calculate the perpendicular distance d to the curve:
\[
d (P, f) = | y - f (x) | \tag {18}
\]
If \( d(P, f) \) exceeds a predetermined threshold \( \delta \), the point \( P \) is considered noise and is removed from the dataset:
\[
\text { If } d (P, f) > \delta , \text { then } P \text { is   noise } \tag {19}
\]
The complete operation process is shown in Algorithm. 1. In order to improve calculation efficiency, we use parallel computing and KDTree's efficient point cloud search scheme. Through this approach, combining the DBSCAN algorithm with polynomial curve fitting effectively identifies and extracts accurate curb lines, while eliminating noise points, thus improving the overall detection accuracy.
Algorithm 1 Multi-Cluster and Curve Fitting Post-Processing Require: Point cloud data, \(\varepsilon\), minPts, \(\delta\)
Ensure: Refined curb line segmentation
1: Step 1: Apply DBSCAN to point cloud data with parallel processing
2: for each point P in point cloud do
3:    Compute  \( N_{\varepsilon}(P) \)  in parallel computation
4:    if  \( |N_{\varepsilon}(P)| \geq \minPts \)  then
5:    Mark P as a core point
6:    end if
7: end for
8: Step 2: Segment curb lines into clusters
9: Step 3: Fit polynomial curves in batch processing
10: for each curb segment do
11:    Fit polynomial curve  \( f(x) \) 
12:    for each point  \( P(x, y) \)  in segment do
13:    Calculate perpendicular distance  \( d(P, f) \) 
14:    Build KDTree for efficient nearest neighbor search
15:    if  \( d(P, f) > \delta \)  then
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
The performance of our curb detection method was rigorously evaluated using standard metrics. These include Precision, Recall, and F-1 score, which are quintessential for quantifying the accuracy and reliability of classification models. Precision, defined as  \( \frac{TP}{TP+FP} \) , measures the proportion of correctly predicted positive observations to the total predicted positives. Recall, calculated as  \( \frac{TP}{TP+FN} \) , assesses the proportion of actual positives that were correctly identified. The F-1 score, given by  \( 2 \times \frac{Precision \times Recall}{Precision + Recall} \) , harmonizes the balance between Precision and Recall, providing a single measure of efficacy. Here, TP (True Positives) represents the number of correct positive predictions, FP (False Positives) denotes the count of negative instances incorrectly classified as positive, TN (True Negatives) refers to the count of correct negative predictions, and FN (False Negatives) signifies the instances where positive cases were wrongly predicted as negative. These metrics collectively offer a comprehensive view of our model's performance, which is crucial for the validation.
B. Quantitative Results of Curb Detection
1) Model Training Results
In the NRS dataset experiments (refer to Table II), this study compared classic segmentation algorithms such as PointPillars, U-Net, Swin-Transformer, and CSWin-Transformer, as well as the state-of-the-art curb detection model, LCDet. Leveraging the specially designed MSCA module for 3D curb scenarios, CurbNet achieved the highest detection performance on the NRS dataset. With the auxiliary training of relevant labels, CurbNet attained Precision, Recall, and F-1 scores of 0.8281, 0.8329, and 0.8308, respectively. Among them, auxiliary training helps improve Precision by 0.5 points, and post-processing helps improve Precision by 1.4 points.
The experiments in the 3D-Curb dataset not only compared classic and advanced deep learning model algorithms but also included three traditional methods of manual feature