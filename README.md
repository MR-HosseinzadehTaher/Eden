# Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision

This repository provides a PyTorch implementation of the [Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision](https://arxiv.org/pdf/2309.15358.pdf).


Existing SSL methods lack capabilities of “understanding” the foundation of medical imaging—human anatomy. We believe that a foundation model must be able to transform each pixel in an image (e.g., a chest X-ray) into semantics-rich numerical vectors, called embeddings, where different anatomical structures (indicated by different colored boxes) are associated with different embeddings, and the same anatomical structures have (nearly) identical embeddings at all resolutions and scales (indicated by different box shapes) across patients. Inspired by the hierarchical nature of human anatomy, we introduce a novel SSL strategy to learn anatomy from medical images, resulting in embeddings **<span style="color:blue">(Eve)</span>**, generated by our pretrained model **<span style="color:blue">(Adam)</span>**, with such desired properties.

<br/>
<p align="center"><img width="100%" src="images/intuition.png" /></p>
<br/>


## Method

Our SSL strategy gradually decomposes and perceives the anatomy in a coarse-to-fine manner. Our **Anatomy Decomposer (AD)** decomposes the anatomy into a hierarchy of parts with granularity level $n \in {0,1,...}$ at each training stage. Thus, anatomical structures of finer-grained granularity will be incrementally presented to the model as the input. Given image $I$, we pass it to AD to get a random anchor $x$. We augment $x$ to generate two views (positive samples), and pass them to two encoders to get their features. To avoid semantic collision in training objective, our **Purposive Pruner** removes semantically similar anatomical structures across images to anchor $x$ from the memory bank. Contrastive loss is then calculated using positive samples’ features and the pruned memory bank. The figure shows pretraining at $n=4$.
<br/>
<p align="center"><img width="100%" src="images/method.png" /></p>
<br/>



## Major results from our work

1. **Adam provides superior performance over fully/self-supervised methods.**
<br/>
<p align="center"><img width="80%" src="images/11.png" /></p>
<br/>

2. **Adam enhances annotation efficiency, revealing promise for fewshot learning.**
<br/>
<p align="center"><img width="90%" src="images/22.png" /></p>
<br/>

3. **Adam preserves locality and compositionality properties, which are intrinsic to anatomical structures and critical for understanding anatomy, in its embedding space.**
<br/>
<p align="center"><img width="90%" src="images/33.png" /></p>
<br/>

4. **Ablation studies on (1) Eve’s accuracy in anatomy understanding, (2) effect of anatomy decomposer, (3) effect of purposive pruner, and (4) adaptability of our framework to other imaging modalities.**
<br/>
<p align="center"><img width="90%" src="images/44.png" /></p>
<br/>

## Installation
Clone the repository and install dependencies using the following command:
```bash
$ git clone https://github.com/MR-HosseinzadehTaher/Eden.git
$ cd Eden-main/
$ pip install -r requirements.txt
```



## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@misc{taher2023foundation,
      title={Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision}, 
      author={Mohammad Reza Hosseinzadeh Taher and Michael B. Gotway and Jianming Liang},
      year={2023},
      eprint={2309.15358},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a
Seed Grant and an Innovation Grant, and in part by the NIH under Award
Number R01HL128785. The content is solely the responsibility of the authors
and does not necessarily represent the official views of the NIH. This work has
utilized the GPUs provided in part by the ASU Research Computing and in
part by the Bridges-2 at Pittsburgh Supercomputing Center through allocation
BCS190015 and the Anvil at Purdue University through allocation MED220025
from the Advanced Cyberinfrastructure Coordination Ecosystem: Services &
Support (ACCESS) program, which is supported by National Science Foundation
grants #2138259, #2138286, #2138307, #2137603, and #2138296. The content
of this paper is covered by patents pending.

