# Awesome-Multi-Task-Learning

Paper List for Multi-Task Learning. (Since 2016)

Conference: ICML, NeurIPS, ICLR, CVPR, ICCV, ECCV, KDD, UAI, ECML PKDD, etc.

Journal: TPAMI, TIP, JMLR, Machine Learning, Artificial Intelligence, etc.

## Architectures for Multi-Task Learning

#### Manually Designed Architectures

- Javaloy, A., & Valera, I. [RotoGrad: Gradient Homogenization in Multitask Learning](https://openreview.net/forum?id=T8wHz4rnuGL "RotoGrad"). ICLR, 2021.

- Hazimeh, H., Zhao, Z., Chowdhery, A., Sathiamoorthy, M., Chen, Y., Mazumder, R., ... & Chi, E. [Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html "DSelect-k"). NeurIPS, 2021.

- Sun, X., Panda, R., Feris, R., & Saenko, K. [Adashare: Learning what to share for efficient deep multi-task learning](https://proceedings.neurips.cc/paper/2020/hash/634841a6831464b64c072c8510c7f35c-Abstract.html "AdaShare"). NeurIPS, 2020.

- Tang, H., Liu, J., Zhao, M., & Gong, X. [Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236?casa_token=6f07DDkXg64AAAAA:D5Yqu4LDFiTrxgOxrFqxa9GyD23wd0aOkUy8ceRo_W-yAYs1qF5jw3iyhOxA7V9YTqFoxBB_j41l "PLE"). RecSys, 2020.

- Cui, C., Shen, Z., Huang, J., Chen, M., Xu, M., Wang, M., & Yin, Y. [Adaptive feature aggregation in deep multi-task convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/944988 "AFANet"). IEEE Transactions on Circuits and Systems for Video Technology, 2019.

- Liu, S., Johns, E., & Davison, A. J. [End-to-end multi-task learning with attention](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html "MTAN"). CVPR, 2019.

- Gao, Y., Ma, J., Zhao, M., Liu, W., & Yuille, A. L. [Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_NDDR-CNN_Layerwise_Feature_Fusing_in_Multi-Task_CNNs_by_Neural_Discriminative_CVPR_2019_paper.html "NDDR-CNN"). CVPR, 2019.

- Bragman, F. J., Tanno, R., Ourselin, S., Alexander, D. C., & Cardoso, J. [Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels](https://openaccess.thecvf.com/content_ICCV_2019/html/Bragman_Stochastic_Filter_Groups_for_Multi-Task_CNNs_Learning_Specialist_and_Generalist_ICCV_2019_paper.html "SFG"). ICCV, 2019.

- Strezoski, G., Noord, N. V., & Worring, M. [Many task learning with task routing](https://openaccess.thecvf.com/content_ICCV_2019/html/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.html "MaTL"). ICCV, 2019.

- Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., & Chi, E. H. [Modeling task relationships in multi-task learning with multi-gate mixture-of-experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007 "MMoE)"). KDD, 2018.

- [**PS-MCNN**] Cao, J., Li, Y., & Zhang, Z. [Partially shared multi-task convolutional neural network with local constraint for face attribute learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Partially_Shared_Multi-Task_CVPR_2018_paper.html "PS-MCNN"). **CVPR**, 2018.

  Notes: the key idea of **PS-MCNN** lies in **sharing a common network for all the groups to learn shared features**, and **constructing group specific network for each group** from the beginning of the architecture to its end to **learn task specific features**; four Task Specific Networks (**TSNets**) and one Shared Network (**SNet**) are connected by Partially Shared (**PS**) structures to learn better **shared** and **task specific** representations.

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

- Sun, G., Probst, T., Paudel, D. P., Popović, N., Kanakis, M., Patel, J., ... & Van Gool, L. [Task Switching Network for Multi-task Learning](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Task_Switching_Network_for_Multi-Task_Learning_ICCV_2021_paper.html "TSNs"). ICCV, 2021.

- Pascal, L., Michiardi, P., Bost, X., Huet, B., & Zuluaga, M. A. [Maximum roaming multi-task learning](https://ojs.aaai.org/index.php/AAAI/article/view/17125 "MR"). AAAI, 2021.

- Guo, P., Lee, C. Y., & Ulbricht, D. [Learning to branch for multi-task learning](https://proceedings.mlr.press/v119/guo20e.html "LearnToBranch"). ICML, 2020.

- Gao, Y., Bai, H., Jie, Z., Ma, J., Jia, K., & Liu, W. [Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_MTL-NAS_Task-Agnostic_Neural_Architecture_Search_Towards_General-Purpose_Multi-Task_Learning_CVPR_2020_paper.html "MTL-NAS"). CVPR, 2020.

- Bruggemann, D., Kanakis, M., Georgoulis, S., & Van Gool, L. [Automated Search for Resource-Efficient Branched Multi-Task Networks](https://www.bmvc2020-conference.com/assets/papers/0359.pdf "BMTAS"). BMVC, 2020.

- [**Routing Network**] Rosenbaum, C., Klinger, T., & Riemer, M. [Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning](https://openreview.net/forum?id=ry8dvM-R- "Routing Network"). **ICLR**, 2018.

  Notes: **Routing** is the process of iteratively applying the **router** to select a sequence of **function blocks** to be composed and applied to the input vector.

- [**Soft order**] Meyerson, E., & Miikkulainen, R. [Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering](https://openreview.net/forum?id=BkXmYfbAZ "Soft order"). **ICLR**, 2018.

  Notes: in the **soft ordering** approach, a joint model learns *how* to **apply shared layers in different ways at different depths for different tasks** as it simultaneously learns the parameters of the layers themselves.

- Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., & Feris, R. [Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification](https://openaccess.thecvf.com/content_cvpr_2017/html/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.html). **CVPR**, 2017.

  Notes: starts with a thin multi-layer network and *dynamically* **widens** it in a greedy manner during training; creates a **tree-like deep architecture**, on which **similar tasks reside in the same branch** until at the top layers.

## Optimization for Multi-Task Learning

#### Optimization Techniques

#### Loss Balancing

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

- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. [Gradient surgery for multi-task learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html "PCGrad"). NeurIPS, 2020.

- [**GradNorm**] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. [Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks](http://proceedings.mlr.press/v80/chen18a.html?ref=https://githubhelp.com "GradNorm"). **ICML**, 2018.

  Notes: automatically balances training in deep multitask models by **dynamically tuning gradient magnitudes**; establish a **common scale** for **gradient magnitudes**, and **balance training rates** of different tasks.
  
  Q: **Does it necessary to make the gradient magnitudes of different tasks to a common scale?** or **Why?**

## Others

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

Please create an issue if you find we missed some papers.