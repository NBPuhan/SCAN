# Stand-alone Composite Attention Network for Concrete Structural Defect Classification
This is the tensorflow implementation of "Stand-alone Composite Attention Network for Concrete Structural Defect Classification". This repository includes the proposed gated fine-grained attention block, bilateral multi-attention module, multi-scale attention module, stem block and the transition unit. The results folder contains the results and the visualizations using attention maps.

## The SCAN architecture
The novel stand-alone composite attention network (SCAN) stems from the novel gated fine-grained attention (GFGA) to capture crucial discriminative local features from visually-similar defect classes, without augmenting with Conv layers. We then propose a multi-scale attention module (MSAM) that encompasses multi-scale information to capture variations in image properties. MSAM incorporates the bilateral multi-attention module (BMAM) to extract salient channel-spatial descriptors. Extensive experimental evaluations along with the ablation study for concrete defect classification on three large datasets show the superiority of our SCAN architecture over other state-of-the-art methodologies.

### The GFGA and BMAM Modules
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Figures/TAI_Blocks_New.png)
The GFGA block is used to select discriminative cues from defect classes such as efflorescence, corrosion and spallation which generally occupy small regions in the image. Inside the GFGA block, the soft-attention mask obtained from nonlinear projection of inputs captures minute defect information to distinguish between defect classes and unwanted artifacts. Moreover, the GFGA blocks regulate the information flow from the previous layer with the efficient gating mechanism to obtain generalized performance beneficial for complex datasets with high variations. On the other hand, BMAM separately investigates the relative importance across spatial and channel dimensions to highlight the relevant regions. In BMAM, we employ spatial attention mechanism using the GFGA blocks to encapsulate variations in spatial planes. Similarly, our proposed channel attention unit designed using GFGA blocks enables the network to focus on discriminatory channels to optimize the classification performance.

### The SCAN architecture, MSAM and Transition module
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Figures/TAI_Revised_New.png)
This figure shows (a). The proposed SCAN. Here Max(a,b) stands for max pooling layer with pool size (a,a) and stride as b, GAP represents the global average pooling layer. In DENSE, two dense layers of 256 and 100 nodes with ReLU activation are used. The numbers associated with GFGA and MSAM modules denote the number of neurons in the dense layer. The Up(2) represents upsampling operation by a factor of 2. (b). The proposed MSAM module. (c). The transition block. In multi-target multi-class classification, structural defects usually occur with diverse ranges in scale, aspect ratio, resolution and the area of appearance. The design of the proposed MSAM is motivated to selectively consolidate multiscale discriminative features, ranging from 48 x 48 to 6 x 6. After reaching the lowest resolution in the respective path, a top-down sequence of upsampling operation is performed followed by the GFGA block to further highlight local information. In each scale, one BMAM module is incorporated for encoding important channel and spatial descriptors to accommodate defect variations. Finally, the features from different scales are aggregated to produce the output with holistic spatial and channel information.

## Hyper-parameters for training

For our results, 200 epochs are considered with mini-batch size of 16 and learning rate 0.001 and momentum 0.9. The data split protocol same as the existing works were being followed.

## Results

### Classification results on CODEBRIM dataset:
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Results/res2_TAI.PNG)
### Classification results on SDNET-2018 dataset:
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Results/res3_TAI.PNG)
### Classification results on Concrete Crack Image dataset:
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Results/res4_TAI.PNG)

### Attention maps using sample images from three datasets:
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Figures/Attention_Maps_TAI.PNG)

These figures show the attention maps obtained from the proposed iDAAM network for sample images from CODEBRIM, concrete crack image and SDNET-2018 datasets and are given in top, middle and bottom row, respectively. Original images are followed by their respective attention maps, placed side-by-side. Here, red color denotes highest attention, while blue denotes the lowest attention. First row from left to right: (a) Exposed bars with spallation and corrosion, (b) Corroded bar with crack and spallation, (c) Spallation with efflorescence, (d) Spalled corroded bar. Second row from left to right: (e) Corroded iron bars with efflorescence, (f) Spalled bar, (g) Corroded bar with crack and spallation, (h) Heavy corrosion. Third row from left to right: (i) Cracked bridge deck image, (j) wall image with crack, (k)-(l) cracked concrete pavement images. Last row from left to right: (m)-(p) Crack defect images from Concrete Crack Image dataset.
