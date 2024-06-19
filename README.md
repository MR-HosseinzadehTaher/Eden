<p align="center"><img width="90%" src="images/Adam_logo.png" /></p>

--------------------------------------------------------------------------------

We devise a novel self-supervised learning (SSL) framework that underpins the development of powerful models foundational to medical imaging via learning anatomy. Our framework not only generates **highly generalizable pretrained models, called Adam (autodidactic
dense anatomical models)**, but also, in contrast to existing SSL methods, yields **dense anatomical embeddings, nicknamed Eve (embedding vectors)**, preserveing a semantic balance of anatomical diversity and harmony, making them semantically meaningful for anatomy understanding.

<p align="center"><img src="images/Adam_Eve.png" /></p>

## Publications

### Representing Part-Whole Hierarchies in Foundation Models by Learning Localizability, Composability, and Decomposability from Anatomy via Self-Supervision

<img alt="Static Badge" src="https://img.shields.io/badge/Adam-Version2-yellow">


:boom: ${\color{red} {\textbf{Accepted at CVPR 2024 [main conference]}}}$

[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
IEEE/CVF Conference on Computer Vision and Pattern Recognition ([CVPR](https://cvpr.thecvf.com/))

<a href='https://arxiv.org/pdf/2404.15672'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='images/Adam_Eve_v2.png'><img src='https://img.shields.io/badge/Poster-PNG-blue'></a> <a href='https://github.com/MR-HosseinzadehTaher/Eden/tree/main/Adam-v2'><img src='https://img.shields.io/badge/Project-Page-Green'></a> [![Oral Presentation](https://badges.aleen42.com/src/youtube.svg)](https:\\youtube_link) 
<br/>

:star: ${\color{blue} {\textbf{Please download the pretrained Adam-v2 PyTorch model as follow. }}}$

| Backbone | #Params. | Download |
|  ----  | ----  |  ----  |
| ConvNeXt-B | 89M | [Link](https://) |

<p align="center"><img src="images/Adam_Eve_v2.png" /></p>

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

### Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision 

<img alt="Static Badge" src="https://img.shields.io/badge/Adam-Version1-blue">

:trophy: ${\color{red} {\textbf{Best Paper Award (Runner-up)}}}$ 

[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2023](https://conferences.miccai.org/2023/en/)); <br/> Domain Adaptation and Representation Transfer <br/>

<a href='https://arxiv.org/pdf/2309.15358.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='https://github.com/MR-HosseinzadehTaher/Eden/tree/main/Adam-v1'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='images/BestPaper.png'><img src='https://img.shields.io/badge/Best%20Paper%20Award%20(Runner%20up)-yellow'></a> [![Oral Presentation](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/1ky57hn0aRg) 
<br/>


:star: ${\color{blue} {\textbf{Please download the pretrained Adam-v1 PyTorch model as follow. }}}$


| Backbone | #Params. | Download |
|  ----  | ----  |  ----  |
| ResNet-50  | 25.6M | [Link](https://docs.google.com/forms/d/e/1FAIpQLSdHcnN6mLUEXebezyQZh3wE3u1RNtBBOpvjbQA8MNXXr9hdHQ/viewform?usp=sf_link) |


## Citation
If you use this code or use our pretrained weights for your research, please cite our paper:
```
@misc{taher2024representing,
      title={Representing Part-Whole Hierarchies in Foundation Models by Learning Localizability, Composability, and Decomposability from Anatomy via Self-Supervision}, 
      author={Mohammad Reza Hosseinzadeh Taher and Michael B. Gotway and Jianming Liang},
      year={2024},
      eprint={2404.15672},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

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

## License

Released under the [ASU GitHub Project License](./LICENSE).

