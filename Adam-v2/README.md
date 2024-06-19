We devise a novel self-supervised learning (SSL) framework that underpins the development of powerful models foundational to medical imaging via learning anatomy. Our framework not only generates **highly generalizable pretrained models, called Adam (autodidactic
dense anatomical models)**, but also, in contrast to existing SSL methods, yields **dense anatomical embeddings, nicknamed Eve (embedding vectors)**, preserveing a semantic balance of anatomical diversity and harmony, making them semantically meaningful for anatomy understanding.

<p align="center"><img src="images/Adam_Eve_2.png" /></p>

## Publication

### Representing Part-Whole Hierarchies in Foundation Models by Learning Localizability, Composability, and Decomposability from Anatomy via Self-Supervision

[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
IEEE/CVF Conference on Computer Vision and Pattern Recognition ([CVPR](https://cvpr.thecvf.com/))

Adam-v2: [Paper](https://arxiv.org/pdf/2404.15672) | [Code](https://github.com/MR-HosseinzadehTaher/Eden/tree/main/Adam-v2) | [Oral Presentation]()
:boom: ${\color{red} {\textbf{Accepted at CVPR 2024 [main conference]}}}$

<br/>

:star: ${\color{blue} {\textbf{Please download the pretrained Adam-v2 PyTorch model as follow. }}}$


| Backbone | #Params. | Download |
|  ----  | ----  |  ----  |
| ConvNeXt-B | 89M | [Link](https://) |

## Requirements
+ Linux
+ Python
+ Install PyTorch ([pytorch.org](http://pytorch.org))


## Installation
Clone the repository and install dependencies using the following command:
```bash
$ git clone https://github.com/MR-HosseinzadehTaher/Eden.git
$ cd Eden-main/Adam-v2
$ pip install -r requirements.txt
```

## Self-supervised pretraining
### 1. Preparing data
For training Adam-v2 base model with ResNet-50 backbone, we used traing set of ChestX-ray14 dataset, which can be downloaded from [this link](https://nihcc.app.box.com/v/ChestXray-NIHCC).

- The downloaded ChestX-ray14 should have a directory structure as follows:
```
ChestX-ray14/
    |--  images/ 
         |-- 00000012_000.png
         |-- 00000017_002.png
         ... 
```
We used the training set based on the official split provided by ChestX-ray14 dataset. Training labels are not used during pretraining stage. The path to images folder is required for pretraining stage and should be updated in the datasets_config.yaml file.

For training Adam-v2 large-scale model with ConvNeXt-B backbone, we used  [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC), [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge),  [VinDr-CXR](https://vindr.ai/datasets/cxr), [Shenzhen](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets), [MIMIC](https://physionet.org/content/mimic-cxr/2.0.0/), [PadChest](), [COVID-19 Radiography Database](), [Indiana ChestX-ray](), [Mendeley-V2](), [COVIDx](), [JSRT](), and [NIH Montgomery](). For using all data for pretraining, the information of these datasets should be added to datasets_config.yaml file, and the name of datasets should be indicated in the training argument "--datasets".

### 2. Pretraining Adam-v2
This implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported. The localizability branch setup follows [DINO](https://github.com/facebookresearch/dino). 

The following is a sample of training Adam-v2 with ResNet-50 backbone using ChestX-ray14 dataset:

```bash
./run.sh
```

## Fine-tuning Adam-v2 on downstream tasks
For classification tasks, a ResNet-50 encoder can be initialized with the pretrained teacher of Adam-v2 as follows:
```python
import torchvision.models as models

num_classes = #number of target task classes
weight = #path to Adam-v2 pretrained model
model = models.__dict__['resnet50'](num_classes=num_classes)
state_dict = torch.load(weight, map_location="cpu")
if "teacher" in state_dict:
   state_dict = state_dict["teacher"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
for k in list(state_dict.keys()):
   if k.startswith('fc'):
      del state_dict[k]
msg = model.load_state_dict(state_dict, strict=False)
print("=> loaded pretrained model '{}'".format(weight))
print("missing keys:", msg.missing_keys)
```

For segmentation tasks, a U-Net can be initialized with the pre-trained encoder of Adam-v2 as follows:
```python
import segmentation_models_pytorch as smp

backbone = 'resnet50'
weight = #path to Adam-v2 pre-trained model
model=smp.Unet(backbone)
state_dict = torch.load(weight, map_location="cpu")
if "state_dict" in state_dict:
   state_dict = state_dict["state_dict"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
for k in list(state_dict.keys()):
   if k.startswith('fc'):
      del state_dict[k]
msg = model.load_state_dict(state_dict, strict=False)
print("=> loaded pre-trained model '{}'".format(weight))
print("missing keys:", msg.missing_keys)

```


## Citation
If you use this code or use our pretrained weights for your research, please cite our paper:
```
@InProceedings{Taher_2024_CVPR,
    author    = {Taher, Mohammad Reza Hosseinzadeh and Gotway, Michael B. and Liang, Jianming},
    title     = {Representing Part-Whole Hierarchies in Foundation Models by Learning Localizability Composability and Decomposability from Anatomy via Self Supervision},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11269-11281}
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

