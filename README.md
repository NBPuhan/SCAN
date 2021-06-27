# Stand-alone Composite Attention Network for Concrete Structural Defect Classification
This is the tensorflow implementation of "Stand-alone Composite Attention Network for Concrete Structural Defect Classification". This repository includes the proposed gated fine-grained attention block, bilateral multi-attention module, multi-scale attention module, stem block and the transition unit. The results folder contains the results and the visualizations using attention maps.

## The SCAN architecture
The novel stand-alone composite attention network (SCAN) stems from the novel gated fine-grained attention (GFGA) to capture crucial discriminative local features from visually-similar defect classes, without augmenting with Conv layers. We then propose a multi-scale attention module (MSAM) that encompasses multi-scale information to capture variations in image properties. MSAM incorporates the bilateral multi-attention module (BMAM) to extract salient channel-spatial descriptors. Extensive experimental evaluations along with the ablation study for concrete defect classification on three large datasets show the superiority of our SCAN architecture over other state-of-the-art methodologies.

### The GFGA and BMAM Modules
![alt text](https://github.com/NBPuhan/SCAN/blob/main/Figures/TAI_Blocks_New.png)
