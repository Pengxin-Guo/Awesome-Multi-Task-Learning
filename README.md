# Awesome-Multi-Task-Learning

Paper List for Multi-Task Learning (focus on architectures and optimization for MTL). (Since 2016)

Contents:

- Architectures for Multi-Task Learning
  - Manually Designed Architectures
  - Learning Architectures
- Optimization for Multi-Task Learning
  - Loss Balancing
  - Gradient Balancing
- Others

## Architectures for Multi-Task Learning

#### Manually Designed Architectures

- [**MulT**] Bhattacharjee, D., Zhang, T., Süsstrunk, S., & Salzmann, M. [MulT: An End-to-End Multitask Learning Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Bhattacharjee_MulT_An_End-to-End_Multitask_Learning_Transformer_CVPR_2022_paper.html "MulT"). **CVPR**, 2022.

  Notes: at the heart of their approach is **a shared attention mechanism** modeling the dependencies across the tasks.

  Q: **How to select the reference task if the surface normal task not in these tasks? Can we automatically choose the reference task?**

- [**Medusa**] Spencer, J., Bowden, R., & Hadfield, S. [Medusa: Universal Feature Learning via Attentional Multitasking](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Spencer_Medusa_Universal_Feature_Learning_via_Attentional_Multitasking_CVPRW_2022_paper.html "Medusa"). **CVPR workshop**, 2022.

  Notes: **shared feature attention** (**spatial attention**) masks relevant backbone features for each task, allowing it to learn a generic representation; a novel **Multi-Scale Attention** head allows the network to better combine per-task features from different scales when making the final prediction.

- [**CA-MTL**] Pilault, J., & Pal, C. [Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data](https://openreview.net/forum?id=de11dbHzAMF "CA-MTL"). **ICLR**, 2021.

  Notes: consisting of a new **conditional attention mechanism** as well as a set of task-conditioned modules (**conditional alignment** and **conditional layer normalization**).

- [**HyperGrid Transformer**] Tay, Y., Zhao, Z., Bahri, D., Metzler, D., & Juan, D. C. [HyperGrid Transformers: Towards A Single Model for Multiple Tasks](https://openreview.net/forum?id=hiq1rHO8pNT "HyperGrid Transformer"). **ICLR**, 2021.

  Notes: leverages **Grid-wise Decomposable Hyper Projections** (HyperGrid), a hypernetwork-based projection layer for task conditioned weight generation.

  Q: Does the task embedding is a learnable vector or a fixed vector?

- [**CTN**] Popovic, N., Paudel, D. P., Probst, T., Sun, G., & Van Gool, L. [Compositetasking: Understanding images by spatial composition of tasks](https://openaccess.thecvf.com/content/CVPR2021/html/Popovic_CompositeTasking_Understanding_Images_by_Spatial_Composition_of_Tasks_CVPR_2021_paper.html "CTN"). **CVPR**, 2021.

  Notes: design **a convolutional neural network that performs multiple, pixelwise tasks**; feed every image along with a composition of spatially distributed multiple task requests to execute pixel-specific tasks; the proposed method for CompositeTasking learns by **task-specific batch normalization**.

- [**ATRC**] Brüggemann, D., Kanakis, M., Obukhov, A., Georgoulis, S., & Van Gool, L. [Exploring relational context for multi-task dense prediction](https://openaccess.thecvf.com/content/ICCV2021/html/Bruggemann_Exploring_Relational_Context_for_Multi-Task_Dense_Prediction_ICCV_2021_paper.html "ATRC"). **ICCV**, 2021.

  Notes: **different source-target task pairs benefit from different context types**; in order to **automate the selection process**, they sample the pool of all available contexts (i.e., *global*, *local*, *T-label*, *S-label*, *none*) for each task pair using **differentiable  NAS techniques**.

- [**TSNs**] Sun, G., Probst, T., Paudel, D. P., Popović, N., Kanakis, M., Patel, J., ... & Van Gool, L. [Task Switching Network for Multi-task Learning](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Task_Switching_Network_for_Multi-Task_Learning_ICCV_2021_paper.html "TSNs"). **ICCV**, 2021.

  Notes: **share all parameters among all tasks** and do not require any task-specific modules (parameters); multiple tasks are performed by switching between them, performing one task at a time.

- [**HGNN**] Guo, P., Deng, C., Xu, L., Huang, X., & Zhang, Y. [Deep multi-task augmented feature learning via hierarchical graph neural network](https://link.springer.com/chapter/10.1007/978-3-030-86486-6_33 "HGNN"). **ECML PKDD**, 2021.

  Notes: HGNN consists of **two-level graph neural networks**; in the low level, an **intra-task GNN** is responsible of learning a powerful representation for each data point in a task by aggregating its neighbors. Based on the learned representation, a **task embedding** can be generated for each task in a similar way to **max pooling**; in the second level, an **inter-task GNN** updates task embeddings of all the tasks based on the **attention mechanism** to model task relations; the task embedding of one task is used to augment the feature representation of data points in this task.

- [**AFANet**] Cui, C., Shen, Z., Huang, J., Chen, M., Xu, M., Wang, M., & Yin, Y. [Adaptive feature aggregation in deep multi-task convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/9449881 "AFANet"). **IEEE Transactions on Circuits and Systems for Video Technology**, 2021.

  An extension of [Deep adaptive feature aggregation in multi-task convolutional neural networks](https://dl.acm.org/doi/abs/10.1145/3340531.3412132?casa_token=52U6Ty7umlIAAAAA:5tnEZP53uqx8uO8w_PsFN3NTY6y0TIFbVzMLJ8mWObw9L6rOGxc4KRLps2jjaQBA4mYtD8zzdhye "AFA"). **CIKM**, 2020.

- [**PSD**] Zhou, L., Cui, Z., Xu, C., Zhang, Z., Wang, C., Zhang, T., & Yang, J. [Pattern-structure diffusion for multi-task learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Pattern-Structure_Diffusion_for_Multi-Task_Learning_CVPR_2020_paper.html "PSD"). **CVPR**, 2020.

  Notes: the **motivation** is **pattern structures high-frequently recur within intra-task also across tasks**; mine and propagate **task-specific and task-across pattern structures** in the task-level space for multiple tasks in two different ways, i.e., **intra-task and inter-task PSD**.

- [**RCM**] Kanakis, M., Bruggemann, D., Saha, S., Georgoulis, S., Obukhov, A., & Gool, L. V. [Reparameterizing convolutions for incremental multi-task learning without task interference](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_41 "RCM"). **ECCV**, 2020.

  Notes: decompose each convolution into **a shared part** that acts as a filter bank encoding common knowledge, and **task-specific modulators** that adapt this common knowledge uniquely for each task; **the shared part is not trainable** to explicitly avoid negative transfer.

- [**MTI-Net**] Vandenhende, S., Georgoulis, S., & Gool, L. V. [Mti-net: Multi-scale task interaction networks for multi-task learning](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_31 "MTI-Net"). **ECCV**, 2020.

  Notes: they show that **two tasks with high affinity at a certain scale are not guaranteed to retain this behaviour at other scales**, and vice versa.

- [**AM-CNN**] Lyu, K., Li, Y., & Zhang, Z. [Attention-aware multi-task convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/8859643 "AM-CNN"). **TIP**, 2020.

  Notes: **automatically learns appropriate sharing** through end-to-end training; the **attention mechanism** (SE block) is introduced into their architecture to **suppress redundant contents contained in the representations**; the **shortcut connection** is adopted to **preserve useful information**.

  Q1: **Which part of their model shows "automatically learns appropriate sharing"?** Is the shared Bottleneck Layer in the AM-CNN module?

  Q2: **Can this method combines with spatial attention to further improve the performance?**

  Q3: They use the one-task networks trained on the corresponding tasks to initialize the networks. This seems to require a lot of training time.

- [**PLE**] Tang, H., Liu, J., Zhao, M., & Gong, X. [Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236?casa_token=6f07DDkXg64AAAAA:D5Yqu4LDFiTrxgOxrFqxa9GyD23wd0aOkUy8ceRo_W-yAYs1qF5jw3iyhOxA7V9YTqFoxBB_j41l "PLE"). **RecSys**, 2020.

  Notes: separate shared components and task-specific components explicitly and adopts a progressive routing mechanism to extract and separate deeper semantic knowledge gradually.

- [**AFA**] Shen, Z., Cui, C., Huang, J., Zong, J., Chen, M., & Yin, Y. [Deep adaptive feature aggregation in multi-task convolutional neural networks](https://dl.acm.org/doi/abs/10.1145/3340531.3412132?casa_token=52U6Ty7umlIAAAAA:5tnEZP53uqx8uO8w_PsFN3NTY6y0TIFbVzMLJ8mWObw9L6rOGxc4KRLps2jjaQBA4mYtD8zzdhye "AFA"). **CIKM**, 2020.

  Notes: a **dynamic aggregation mechanism** is designed to **allow each task to adaptively determine the degree to which the feature aggregation of different tasks is needed according to the feature dependencies**; the AFA consists of two main ingredients: the **Channel-wise Aggregation
  Module** (CAM) and **Spatial-wise Aggregation Module** (SAM).

- [**NDDR-CNN**, **NDDR-CNN-Shortcut**] Gao, Y., Ma, J., Zhao, M., Liu, W., & Yuille, A. L. [Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_NDDR-CNN_Layerwise_Feature_Fusing_in_Multi-Task_CNNs_by_Neural_Discriminative_CVPR_2019_paper.html "NDDR-CNN, NDDR-CNN-Shortcut"). **CVPR**, 2019.

  Motivation: **Why would we assume that the low- and mid-level features for different tasks in MTL should be identical, especially when the tasks are loosely related? If not, is it optimal to share the features until the last convolutional layer?**

  Notes: they hypothesize that **these features, obtained from multiple feature descriptors (i.e., different CNN levels from multiple tasks), contain additional discriminative information of input data, which should be exploited in MTL towards better performance**; they **concatenate** all the task-specific features with the same spatial resolution from different tasks according to the feature channel dimension, conduct **discriminative dimensionality reduction** on the concatenated features.

- [**MTAN**, **DWA**] Liu, S., Johns, E., & Davison, A. J. [End-to-end multi-task learning with attention](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html "MTAN, DWA"). **CVPR**, 2019.

  Notes: consists of a single shared network containing a global feature pool, together with a **soft-attention module for each task**; Dynamic Weight Average (**DWA**), which adapts the task weighting over time by considering the rate of change of the loss for each task.

  Q: **Can we replace the $K$ task-specific attention networks with one task-specific attention network to further reduce the parameters?** This task-specific attention network can receive the task embeddings and the shared features as input.

- [**PAP**] Zhang, Z., Cui, Z., Xu, C., Yan, Y., Sebe, N., & Yang, J. [Pattern-affinitive propagation across depth, surface normal and semantic segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.html "PAP"). **CVPR**, 2019.

  Notes: the **motivation** behind it comes from the statistic observation that **pattern-affinitive pairs recur much frequently across different tasks as well as within a task**; conduct two types of propagations, **cross-task propagation** and **task-specific propagation**, to **adaptively diffuse those similar patterns**.

  Q: **What is the finally learned ![](http://latex.codecogs.com/svg.latex?\alpha)?** Since it can represents the task relationship to some extent.

- [**ASTMT**] Maninis, K. K., Radosavovic, I., & Kokkinos, I. [Attentive single-tasking of multiple tasks](https://openaccess.thecvf.com/content_CVPR_2019/html/Maninis_Attentive_Single-Tasking_of_Multiple_Tasks_CVPR_2019_paper.html "ASTMT"). **CVPR**, 2019.

  Notes: a network is trained on multiple tasks, but performs one task at a time; use **data-dependent modulation signals** (task-specific SE block) that enhance or suppress neuronal activity in a task-specific manner; use **task-specific Residual Adapter blocks** that latch on to a larger architecture in order to extract task-specific information which is fused with the representations extracted by a generic backbone; reduce task interference by **forcing the task gradients to be statistically indistinguishable through adversarial training**, ensuring that the common backbone architecture serving all tasks is not dominated by any of the task-specific gradients.

- [**MRAN**] Zhao, J., Du, B., Sun, L., Zhuang, F., Lv, W., & Xiong, H. [Multiple relational attention network for multi-task learning](https://dl.acm.org/doi/abs/10.1145/3292500.3330861?casa_token=OWmM19esGkgAAAAA:VcB0sDcRzF_7Sqn6C-F0zAIfgmLUvA-UISFEG3jrdpgcqLT1tTIaJQwKH83jzTGo7tWQcPpfxGgo "MRAN"). **KDD**, 2019.

  Notes: consists of **three attention-based relationship learning modules**: **task-task relationship**, **feature-feature interaction**, and **task-feature dependence**.

- [**MT-DNN**] Liu, X., He, P., Chen, W., & Gao, J. [Multi-task deep neural networks for natural language understanding](https://aclanthology.org/P19-1441/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter "MT-DNN"). **ACL**, 2019.

  Notes: a hard parameter sharing architecture for NLP.

- [**PS-MCNN**] Cao, J., Li, Y., & Zhang, Z. [Partially shared multi-task convolutional neural network with local constraint for face attribute learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Partially_Shared_Multi-Task_CVPR_2018_paper.html "PS-MCNN"). **CVPR**, 2018.

  Notes: the key idea of **PS-MCNN** lies in **sharing a common network for all the groups to learn shared features**, and **constructing group specific network for each group** from the beginning of the architecture to its end to **learn task specific features**; four Task Specific Networks (**TSNets**) and one Shared Network (**SNet**) are connected by Partially Shared (**PS**) structures to learn better **shared** and **task specific** representations.

- [**PAD-Net**] Xu, D., Ouyang, W., Wang, X., & Sebe, N. [Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html "PAD-Net"). **CVPR**, 2018.

  Notes: produces a set of **intermediate auxiliary tasks** providing rich multi-modal data for learning the target tasks; design and investigate three different **multi-modal distillation** modules for deep multi-modal data fusion.

- [**MMoE**] Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., & Chi, E. H. [Modeling task relationships in multi-task learning with multi-gate mixture-of-experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007 "MMoE"). **KDD**, 2018.

  Notes: adapt the Mixture-of-Experts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks, while also having a **gating network** trained to optimize each task; the gating networks take the input features and **output softmax gates assembling the experts with different weights**, allowing different tasks to utilize experts differently.

  Q1: **How to determine the number of experts? Is it same to the task number?**

  Q2: **What is the architecture of each expert?** Is each expert same to the shared bottom? If this, the model parameters of MMoE will become so large.

  Q3: **Why the shared-bottom method is still inferior to MMoE on two identical tasks?** (Figure 4(c)) **Does it means even for two totally relevant tasks, we still cannot share all the parameters in the bottom layers?**

- [**UberNet**] Kokkinos, I. [Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory](https://openaccess.thecvf.com/content_cvpr_2017/html/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.html "UberNet"). **CVPR**, 2017.

  Notes: an **image pyramid** is formed by **successive down-sampling** operations, and each image is processed by a CNN with tied weights.

- [**multinet**] Bilen, H., & Vedaldi, A. [Integrated perception with recurrent multi-task neural networks](https://proceedings.neurips.cc/paper/2016/hash/06409663226af2f3114485aa4e0a23b4-Abstract.html "multinet"). **NeurIPS**, 2016.

  Notes: not only **deep image features are shared** between tasks, but where tasks can interact in a **recurrent** manner by encoding the results of their analysis in a common shared representation of the data.

- [**Cross-stitch**] Misra, I., Shrivastava, A., Gupta, A., & Hebert, M. [Cross-stitch networks for multi-task learning](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html "Cross-stitch"). **CVPR**, 2016.

  Notes:  **automatically** learn an optimal combination of **shared** and **task-specific** representations.

- [**MNC**] Dai, J., He, K., & Sun, J. [Instance-aware semantic segmentation via multi-task network cascades](https://openaccess.thecvf.com/content_cvpr_2016/html/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.html "MNC"). **CVPR**, 2016.

  Notes: decompose the **instance-aware semantic segmentation** task into three different and related sub-tasks; expect that each sub-task is **simpler** than the original instance segmentation task, and is **more easily** addressed by convolutional networks; their network **cascades** have three stages, each of which addresses one sub-task.

- [**FCNN**] Li, X., Zhao, L., Wei, L., Yang, M. H., Wu, F., Zhuang, Y., ... & Wang, J. [Deepsaliency: Multi-task deep neural network model for salient object detection](https://ieeexplore.ieee.org/abstract/document/7488288 "FCNN"). **TIP**, 2016.

  Notes: carry out the task of **saliency detection** in conjunction with the task of **object class segmentation**, which **share a convolution part** with 15 layers; use two networks performing the two tasks by sharing features, which forms a **tree structured network architecture**.

#### Learning Architectures

- [**LearnToBranch**] Guo, P., Lee, C. Y., & Ulbricht, D. [Learning to branch for multi-task learning](https://proceedings.mlr.press/v119/guo20e.html "LearnToBranch"). **ICML**, 2020.

  Notes: propose a **tree-structured network design space** that can **automatically learn how to branch a network**.

  Q: **How to decide the number of child nodes?**

- [**AdaShare**] Sun, X., Panda, R., Feris, R., & Saenko, K. [Adashare: Learning what to share for efficient deep multi-task learning](https://proceedings.neurips.cc/paper/2020/hash/634841a6831464b64c072c8510c7f35c-Abstract.html "AdaShare"). **NeurIPS**, 2020.

  Notes: learn the sharing pattern through a task-specific policy that **selectively chooses which layers to execute for a given task** in the multi-task network.

  Q1: It has to perform ![](http://latex.codecogs.com/svg.latex?T) times forward pass since each task has different paths, which is time-consuming.

  Q2: Will it be effective to extend *AdaShare* for finding a fine-grained **channel sharing pattern** across tasks?

- [**MTL-NAS**] Gao, Y., Bai, H., Jie, Z., Ma, J., Jia, K., & Liu, W. [Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_MTL-NAS_Task-Agnostic_Neural_Architecture_Search_Towards_General-Purpose_Multi-Task_Learning_CVPR_2020_paper.html "MTL-NAS"). **CVPR**, 2020.

  Notes: **disentangle the GP-MTL networks into single-task backbones, and a hierarchical and layerwise features sharing/fusing scheme across them**; fix the single-task backbone branches and search good inter-task edges for hierarchical and layerwise feature fusion/embedding.

  Q: **How can this method generalize to multiple task (i.e., task numbers greater 3)?**

- [**BMTAS**] Bruggemann, D., Kanakis, M., Georgoulis, S., & Van Gool, L. [Automated Search for Resource-Efficient Branched Multi-Task Networks](https://www.bmvc2020-conference.com/assets/papers/0359.pdf "BMTAS"). **BMVC**, 2020.

  Notes: automatically define branching (tree-like) structures in the encoding stage of a multi-task neural network.

- [**DEN**] Ahn, C., Kim, E., & Oh, S. [Deep elastic networks with model selection for multi-task learning](https://openaccess.thecvf.com/content_ICCV_2019/html/Ahn_Deep_Elastic_Networks_With_Model_Selection_for_Multi-Task_Learning_ICCV_2019_paper.html "DEN"). **ICCV**, 2019.

  Notes: the proposed method consists of an **estimator** and a **selector**, the estimator can produce multiple different network models of different configurations in a hierarchical structure, the selector chooses a model dynamically from a pool of candidate models given an input instance.

- [**SFG**] Bragman, F. J., Tanno, R., Ourselin, S., Alexander, D. C., & Cardoso, J. [Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels](https://openaccess.thecvf.com/content_ICCV_2019/html/Bragman_Stochastic_Filter_Groups_for_Multi-Task_CNNs_Learning_Specialist_and_Generalist_ICCV_2019_paper.html "SFG"). **ICCV**, 2019.

  Notes: the **SFGs** learns to **allocate kernels in each convolution layer into either ''specialist'' groups or a ''shared'' trunk**, which are specific to or shared across different tasks, respectively.

- [**Routing Network**] Rosenbaum, C., Klinger, T., & Riemer, M. [Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning](https://openreview.net/forum?id=ry8dvM-R- "Routing Network"). **ICLR**, 2018.

  Notes: **Routing** is the process of iteratively applying the **router** to select a sequence of **function blocks** to be composed and applied to the input vector.

- [**Soft order**] Meyerson, E., & Miikkulainen, R. [Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering](https://openreview.net/forum?id=BkXmYfbAZ "Soft order"). **ICLR**, 2018.

  Notes: in the **soft ordering** approach, a joint model learns *how* to **apply shared layers in different ways at different depths for different tasks** as it simultaneously learns the parameters of the layers themselves.

- [**FAFS**] Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., & Feris, R. [Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification](https://openaccess.thecvf.com/content_cvpr_2017/html/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.html "FAFS"). **CVPR**, 2017.

  Notes: starts with a thin multi-layer network and *dynamically* **widens** it in a greedy manner during training; creates a **tree-like deep architecture**, on which **similar tasks reside in the same branch** until at the top layers.

## Optimization for Multi-Task Learning

#### Loss Balancing

- [**LSB**] Lee, J. H., Lee, C., & Kim, C. S. [Learning Multiple Pixelwise Tasks Based on Loss Scale Balancing](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_Learning_Multiple_Pixelwise_Tasks_Based_on_Loss_Scale_Balancing_ICCV_2021_paper.html "LSB"). **ICCV**, 2021.

  Notes: dynamically adjusts the linear weights to learn all tasks effectively by **balancing the loss scale** (**the product of the loss value and its weight**) periodically.
  
  Q: Does 3rd period is useful? Since 2nd period  has achieved that all tasks are learned at a similar pace, which means the difficulty of all task are equal according to the measurement of Eq.(9).
  
- [**MultiNet++**, **GLS**] Chennupati, S., Sistu, G., Yogamani, S., & A Rawashdeh, S. [Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning](https://openaccess.thecvf.com/content_CVPRW_2019/html/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.html "MultiNet++, GLS"). **CVPR workshop**, 2019.

  Notes: they propose a **multi-stream multi-task network** to take advantage of using feature representations from **preceding frames in a video sequence** for joint learning of segmentation, depth, and motion; they express the total loss of a multi-task learning problem as **geometric mean** of individual task losses, they refer to this as Geometric Loss Strategy (**GLS**).
  
- [**Uncertainty**] Kendall, A., Gal, Y., & Cipolla, R. [Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html "Uncertainty"). **CVPR**, 2018.

  Notes: weighs multiple loss functions by considering the **homoscedastic uncertainty** of each task.
  
  Q1:  Their results show that there is usually not a single optimal weighting for all tasks. Therefore, **what is the optimal weighting?**  Is multi-task learning is an ill-posed optimisation problem without a single higher-level goal?
  
  Q2: Why do the semantics and depth tasks outperform the semantics and instance tasks results? Clearly the three tasks explored in this paper are complimentary and useful for learning a rich representation about the scene. **But can we quantify relationships between tasks?**
  
  Q3: **Where the optimal location is for splitting the shared encoder network into separate decoders for each task?** And, what network depth is best for the shared multi-task representation?
  
- [**Dynamic**] Guo, M., Haque, A., Huang, D. A., Yeung, S., & Fei-Fei, L. [Dynamic task prioritization for multitask learning](https://openaccess.thecvf.com/content_ECCV_2018/html/Michelle_Guo_Focus_on_the_ECCV_2018_paper.html "Dynamic"). **ECCV**, 2018.

  Notes: automatically **prioritize more difficult tasks** by adaptively **adjusting the mixing weight of each task’s loss objective**.

- [**SPMTL**] Li, C., Yan, J., Wei, F., Dong, W., Liu, Q., & Zha, H. [Self-paced multi-task learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14535/14390 "SPMTL"). **AAAI**, 2017.

  Notes: learn the multi-task model in a **self-paced** regime, **from easy instances and tasks to hard instances and tasks**.

- [**SMTL**, **OSMTL**] Murugesan, K., Liu, H., Carbonell, J., & Yang, Y. [Adaptive smoothed online multi-task learning](https://proceedings.neurips.cc/paper/2016/hash/a869ccbcbd9568808b8497e28275c7c8-Abstract.html). **NeurIPS**, 2016.

  Notes: *efficiently* learns multiple related tasks by estimating the **task relationship matrix** from the data; maybe can be formulated as an end-to-end training procedure.

#### Gradient Balancing

- [**Nash-MTL**] Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E. [Multi-task learning as a bargaining game](https://proceedings.mlr.press/v162/navon22a.html "Nash-MTL"). **ICML**, 2022.

  Notes: frame the gradient combination step in MTL as a **bargaining game** and use the **Nash bargaining solution** to find the optimal update direction.

- [**Seq.Reptile**] Lee, S., Lee, H., Lee, J., & Hwang, S. J. [Sequential Reptile: Inter-Task Gradient Alignment for Multilingual Learning](https://openreview.net/forum?id=ivQruZvXxtz "Seq.Reptile"). **ICLR**, 2022.

  Notes: they want their model to maximally retain the knowledge of the pretrained model by finding a good trade-off between **minimizing the downstream MTL loss** and **maximizing the cosine similarity between the task gradients**; in order to **consider gradient alignment across tasks** as well, they propose to let the inner-learning trajectory consist of mini-batches **randomly sampled from all tasks**, which they call **Sequential Reptile**.

  Q: Where do they demonstrate "it is crucial for those tasks to align gradients between them in order to maximize knowledge transfer while minimizing negative transfer"? In other world, **is it helpful for model learning to increase the cosine similarity between the task gradients?**

- [**RotoGrad**] Javaloy, A., & Valera, I. [RotoGrad: Gradient Homogenization in Multitask Learning](https://openreview.net/forum?id=T8wHz4rnuGL "RotoGrad"). **ICLR**, 2022.

  Notes: **jointly homogenize gradient magnitudes and directions**; address the **magnitude differences** by **re-weighting task gradients** at each step of the learning, while **encouraging learning those tasks that have converged the least thus far**; address the **conflicting directions** by smoothly **rotating the shared feature space differently for each task**, seamlessly aligning gradients in the long run.

- [**CAGrad**] Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. [Conflict-averse gradient descent for multi-task learning](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html "CAGrad"). **NeurIPS**, 2021.

  Notes: minimizes the average loss function, while **leveraging the worst local improvement of individual tasks to regularize the algorithm trajectory**.

  Q: **Does the largest loss will dominate the update?**  Since the Eq.(2) is based on the absolute value of the loss.

- [**IMTL, IMTL-G, IMTL-L**] Liu, L., Li, Y., Kuang, Z., Xue, J. H., Chen, Y., Yang, W., ... & Zhang, W. [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr "IMTL"). **ICLR**, 2021.

  Notes: **IMTL-G(rad)**, learn the **scaling factors** such that **the aggregated gradient of task-shared parameters has equal projections onto the raw gradients of individual tasks**; **IMTL-L(oss)**, automatically **learn a loss weighting parameter for each task** so that **the weighted losses have comparable scales** and the effect of different loss scales from various tasks can be canceled-out.

  Q1: **What is the benefit of "the aggregated gradient of task-shared parameters has equal projections onto the raw gradients of individual tasks"?** or **Why to do this?**

  Q2: **Can we use different learning rate for different tasks to achieve the same goal of IMTL-G** (treat all tasks equally so that they progress in the same speed and none is left behind)?

- [**GradVac**] Wang, Z., Tsvetkov, Y., Firat, O., & Cao, Y. [Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models](https://openreview.net/forum?id=F1vEjWK-lH_ "GradVac"). **ICLR**, 2021.

  Notes: relax the assumption of **PCGrad** (any two tasks must have the same gradient similarity objective of zero), set **adaptive gradient similarity objectives**; operate on both **negative and positive gradient similarity**.

  Q: **Is it necessary to operate on the positive gradient similarity?**

- [**PCGrad**] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. [Gradient surgery for multi-task learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html "PCGrad"). **NeurIPS**, 2020.

  Notes: projects a task’s gradient onto the normal plane of the gradient of any other task that has a **conflicting** gradient.

  Q1: **Does the Theorem 2 in their paper means that PCGrad only works under such three conditions, what if the three conditions are not satisfied?**

  Q2: **Is the *tragic triad* of multi-task learning a major factor in making optimization for multi-task learning challenging?** They seem do not answer this question in their experiments.

- [**GradDrop**] Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., & Anguelov, D. [Just pick a sign: Optimizing deep multitask models with gradient sign dropout](https://proceedings.neurips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html "GradDrop"). **NeurIPS**, 2020.

  Notes: select one sign (positive or negative) based on the distribution of gradient values, and **mask out** all gradient values of the opposite sign.

  Q: About the formulation of Gradient Positive Sign Purity, **what if the gradients not in an order of magnitude?** The largest gradient may dominate the sign.

- [**GradNorm**] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. [Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks](http://proceedings.mlr.press/v80/chen18a.html?ref=https://githubhelp.com "GradNorm"). **ICML**, 2018.

  Notes: automatically balances training in deep multitask models by **dynamically tuning gradient magnitudes**; establish a **common scale** for **gradient magnitudes**, and **balance training rates** of different tasks.
  
  Q: **Does it necessary to make the gradient magnitudes of different tasks to a common scale?** or **Why?**
  
- [**MGDA-UB**] Sener, O., & Koltun, V. [Multi-task learning as multi-objective optimization](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html "MGDA-UB"). **NeurIPS**, 2018.

  Notes: cast multi-task learning as **multi-objective optimization**; provide **an upper bound** for the multiple-gradient descent algorithm (**MGDA**) optimization objective and show that it can be computed via **a single backward pass**.

  Q: **Why the derivate of the representations can be computed in a single backward pass, while for the shared parameter needs ![](http://latex.codecogs.com/svg.latex?T) times?**

- [**Modulation**] Zhao, X., Li, H., Shen, X., Liang, X., & Wu, Y. [A modulation module for multi-task learning with applications in image retrieval](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiangyun_Zhao_A_Modulation_Module_ECCV_2018_paper.html "Modulation"). **ECCV**, 2018.

  Notes: propose a general **modulation module**, which can be inserted into any convolutional neural network architecture, to encourage the **coupling and feature sharing of relevant tasks** while **disentangling the learning of irrelevant tasks** with minor parameters addition; equipped with this module, **gradient directions from different tasks can be enforced to be consistent for those shared parameters**.

  Q1: Is it harmful if we force the gradient direction of **irrelevant tasks** to be consistent with the shared parameters? Since the objective of this work is to make the gradient directions of different tasks to be consistent, does it will be worse if the gradient direction of **irrelevant tasks** to be consistent?

  Q2: Why "Since the task-specific masks/projection matrices are **learnable**, we observe that the
  training process will *naturally* mitigate the destructive interference by reducing the average across-task gradient angles" ? **What term in the loss function guide this phenomenon?**

  Q3: They simply apply a **channel-wise scaling vector** in their method, what will happen if we use **both channel attention and spatial attention**?

## Others

- [**SRDML**] Bai, G., & Zhao, L. [Saliency-Regularized Deep Multi-Task Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539442?casa_token=yeg0_7ssl1IAAAAA:i-TgT5J_YELpo749FRj2Sp0pPXk8B906sYlyYWj_DC6GVt5ECfmiXQxhLnHcsrTyY6MGGIG6B1c "SRDML"). **KDD**, 2022.

  Notes: **model the task relation as the similarity between tasks’ input gradients**.

  Q1: **Why tasks’ input gradients can be regard as an measurement of task relations?**

  Q2: **It seems only can be applied to classification tasks**.

- [**TAG**] Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., & Finn, C. [Efficiently identifying task groupings for multi-task learning](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html "TAG"). **NeurIPS**, 2021.

  Notes: suggest a measure of **inter-task affinity** that can be used to systematically and efficiently determine **task groupings** for multi-task learning; measure inter-task affinity by training all tasks together in a single multi-task network and **quantifying the effect to which one task’s gradient update would affect another task’s loss**.

  Finding: **the relationships among tasks change throughout training as measured by inter-task affinity** (this maybe because the different data in different training step); **how tasks should be trained together does not simply depend on the relationships among tasks, but also on detailed aspects of the model and training**.

  Q: "Approximating higher-order affinity scores for each network consisting of three or more tasks" is **time-consuming** especially when the task number is large.

- [**DSelect-k**] Hazimeh, H., Zhao, Z., Chowdhery, A., Sathiamoorthy, M., Chen, Y., Mazumder, R., ... & Chi, E. [Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html "DSelect-k"). **NeurIPS**, 2021.

  Notes: a continuously differentiable and sparse gate for MoE, based on a novel binary encoding formulation.

- [**GTTN**] Zhang, Y., Zhang, Y., & Wang, W. [Multi-task learning via generalized tensor trace norm](https://dl.acm.org/doi/pdf/10.1145/3447548.3467329?casa_token=nH8nZoAxxJsAAAAA:I7BjwUUmiv5uW8hc7r9QZ1ydomGtmFfVh8-saZm9Wc_Ci70qBUZqB54pps3kUJF9EGW1nCkoaHZ3 "GTTN"). **KDD**, 2021.

  Notes: exploits all possible tensor flattenings and it is **defined as the convex sum of matrix trace
  norms of all possible tensor flattenings**.

- [**TRL**] Strezoski, G., Noord, N. V., & Worring, M. [Many task learning with task routing](https://openaccess.thecvf.com/content_ICCV_2019/html/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.html "TRL"). **ICCV**, 2019.

  Notes: introduce a **task-routing** mechanism allowing tasks to have separate in-model data flows; apply a **channel-wise task-specific binary mask** over the convolutional activations, **the masks are generated randomly and kept constant**.

  Q: What if the randomly generated masks is not suitable? **Is it better to make the masks learnable parameters?**

- [**RCMTL**] Yao, Y., Cao, J., & Chen, H. [Robust task grouping with representative tasks for clustered multi-task learning](https://dl.acm.org/doi/abs/10.1145/3292500.3330904?casa_token=sAqWEl99yEIAAAAA:vts_RUH_RXR-sFrf_4BRV7zfm5RKrb4mcE1w6V9Rz_8a0RK9VffbHTTqWW7uY39AzhktdbBRyXF9_VM "RCMTL"). **KDD**, 2019.

  Notes: select a subset of tasks that can represent the whole tasks, all tasks can be clustered into different groups based on these **representative tasks**.

  :warning: Very similar to [Flexible clustered multi-task learning by learning representative tasks](https://ieeexplore.ieee.org/abstract/document/7150415 "FCMTL"). **TPAMI**, 2016.

- [**AMTFL**, **Deep-AMTFL**]Lee, H. B., Yang, E., & Hwang, S. J. [Deep asymmetric multi-task feature learning](http://proceedings.mlr.press/v80/lee18d.html "AMTFL, Deep-AMTFL"). **ICML**, 2018.

  Notes: introduce an **asymmetric autoencoder** term that allows reliable predictors for the easy
  tasks to have high contribution to the **feature learning** while suppressing the influences of unreliable predictors for more difficult tasks; the reconstruction is done in the **feature space** and in a **non-linear** manner.

- [**MRN**] Long, M., Cao, Z., Wang, J., & Yu, P. S. [Learning multiple tasks with multilinear relationship networks](https://proceedings.neurips.cc/paper/2017/hash/03e0704b5690a2dee1861dc3ad3316c9-Abstract.html "MRN"). **NeurIPS**, 2017.

  Notes: mainly based on **tensor normal distributions**; jointly learning **transferable features** and **multilinear relationships**, alleviate the dilemma of **negative-transfer** in feature layers and **under-transfer** in classifier layer.

- [**DMTRL**] Yang, Y., & Hospedales, T. M. [Deep Multi-task Representation Learning: A Tensor Factorisation Approach](https://openreview.net/forum?id=SkhU2fcll "DMTRL"). **ICLR**, 2017.

  Notes: generalise matrix factorisation-based multi-task ideas to **tensor factorisation**; **Tucker Decomposition**; **Tensor Train (TT) Decomposition**.

- [**TNRDMTL**] Yang, Y., & Hospedales, T. M. [Trace Norm Regularised Deep Multi-Task Learning](https://openreview.net/forum?id=rknkNR7Ke "TNRDMTL"). **ICLR workshop**, 2017.

  Notes: the parameters from all models are regularised by the **tensor trace norm**.

- [**CILICIA**] Sarafianos, N., Giannakopoulos, T., Nikou, C., & Kakadiaris, I. A. [Curriculum learning for multi-task classification of visual attributes](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w38/html/Sarafianos_Curriculum_Learning_for_ICCV_2017_paper.html "CILICIA"). **ICCV workshop**, 2017.

  Notes: individual tasks are grouped based on their correlation so that **two groups of strongly and weakly correlated tasks** are formed; the two groups of tasks are learned in a **curriculum learning** setup by transferring the acquired knowledge **from the strongly to the weakly correlated**; the learning process within each group is performed in a **multi-task** classification setup.

- [**HC-MTL**] Liu, A. A., Su, Y. T., Nie, W. Z., & Kankanhalli, M. [Hierarchical clustering multi-task learning for joint human action grouping and recognition](https://ieeexplore.ieee.org/abstract/document/7423818 "HC-MTL"). **TPAMI**, 2017.

  Notes: formulate the objective function with two latent variables, **model parameters** and **grouping information**, for joint optimization; decompose it into two sub-tasks, **multi-task learning** and **task relatedness discovery**, and **iteratively** solve the two sub-tasks.

- [**HD-MTL**] Fan, J., Zhao, T., Kuang, Z., Zheng, Y., Zhang, J., Yu, J., & Peng, J. [HD-MTL: Hierarchical deep multi-task learning for large-scale visual recognition](https://ieeexplore.ieee.org/abstract/document/7849143 "HD-MTL"). **TIP**, 2017.

  Notes: **multiple sets of deep features** are extracted from the **different layers**; a **visual tree** is learned by assigning the visually-similar atomic object classes with similar learning complexities into the same group, it can provide a good environment for **identifying the inter-related learning tasks** automatically. 

- [**AMTL**] Lee, G., Yang, E., & Hwang, S. [Asymmetric multi-task learning based on task relatedness and loss](https://proceedings.mlr.press/v48/leeb16.html "AMTL"). **ICML**, 2016.

  Notes: avoid **negative transfer** problem; allow for **asymmetric information transfer** between the tasks.

  Assumption: each underlying **model parameter** is *succinctly* represented by the **linear combination** of other parameters.

- [**MITL**] Lin, K., Xu, J., Baytas, I. M., Ji, S., & Zhou, J. [Multi-task feature interaction learning](https://dl.acm.org/doi/abs/10.1145/2939672.2939834 "MITL"). **KDD**, 2016.

  Notes: exploit task relatedness in the form of shared representations in both the **original input space** and the **interaction space among features**.

- [**FCMTL**] Zhou, Q., & Zhao, Q. [Flexible clustered multi-task learning by learning representative tasks](https://ieeexplore.ieee.org/abstract/document/7150415 "FCMTL"). **TPAMI**, 2016.

  Notes: a subset of tasks (**representative tasks**) in multi-task learning can be used to represent other tasks due to the similarity among multiple tasks; allow **one task to be clustered into multiple clusters**, and with different weights.

------

Please create an issue or contact 12032913@mail.sustech.edu.cn if you find we missed some papers.