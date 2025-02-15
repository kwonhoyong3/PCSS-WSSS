# PCSS-WSSS
### Official repository for ECCV 2024 paper: [Phase Concentration and Shortcut Suppression for Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04729.pdf) by Hoyong Kwon, Jaeseok Jeong, Sung-Hoon Yoon, and Kuk-Jin Yoon.
---

# 1.Prerequisite
## 1.1 Environment
* Our experiments are worked on Python 3.9, PyTorch 2.0.1, CUDA 11.7, and TITAN RTX.

* You can create conda environment with the provided yaml file.
```
conda env create -f environment.yaml
```
* If you got "TypeError: init() got an unexpected keyword argument 'pretrained_cfg'" error, remove the 'pretrained_cfg' parameter where the error occurs. It will handle the issue.

## 1.2 Dataset Preparation
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
- Download MS COCO images from the official COCO website [here](https://cocodataset.org/#download).
- Download semantic segmentation annotations for the MS COCO dataset [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing). (Refer [RIB](https://github.com/jbeomlee93/RIB))

- Directory hierarchy 
```
    ./data
    ├── VOC2012       
    └── COCO2014            
            ├── SegmentationClass     # GT dir             
            ├── train2014  # train images downloaded from the official COCO website 
            └── val2014    # val images downloaded from the official COCO website
```



* Download the ImageNet-pretrained DeiT-S model from [here](https://github.com/facebookresearch/deit). 
**You need to place the weights as "./pretrained/deit_small_patch16_224-cd65a155.pth. "**

# 2. Usage
> With the following code, you can generate CAMs (seeds) to train the segmentation network.
> For the further refinement, refer [PSA](https://github.com/jiwoon-ahn/psa). 


## 2.1 Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]

For PASCAL:
```
python train_PCSS.py --name [exp_name]
```
For COCO:
```
python train_PCSS_coco.py --name [exp_name]
```

**Note that the mIoU in COCO training set is evaluated on the subset (5.2k images, not the full set of 80k images) for fast evaluation**

## 2.2 Inference (CAM)
* Pretrained weight (PASCAL, seed: 69.5% mIoU) can be downloaded [here](https://drive.google.com/drive/folders/1fL4ntT3FiomuzgBHT67gz9xgiBPPxfgy?usp=sharing) (eccv24_pcss_wsss_69.5_pascal.pth).

For pretrained model (69.5%):
```
python infer_PCSS.py --name [exp_name] --load_pretrained [path_to_ckpt] --dict
```

For model you trained:

```
python infer_PCSS.py --name [exp_name] --load_epo [EPOCH] --dict
```

## 2.3 Evaluation (CAM)
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```


# 3. Additional Information
## 3.1 Paper citation
If our work is helpful for your research, please consider citing our ECCV 2024 paper using the following BibTeX entry.
```
@inproceedings{kwon2024phase,
  title={Phase Concentration and Shortcut Suppression for Weakly Supervised Semantic Segmentation},
  author={Kwon, Hoyong and Jeong, Jaeseok and Yoon, Sung-Hoon and Yoon, Kuk-Jin},
  booktitle={European Conference on Computer Vision},
  pages={293--312},
  year={2024},
  organization={Springer}
}
```
You can also check our earlier works published on ICCV 2021 ([OC-CSE](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)) , ECCV 2022 ([AEFT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf)), CVPR 2023 ([ACR](https://openaccess.thecvf.com/content/CVPR2023/papers/Kweon_Weakly_Supervised_Semantic_Segmentation_via_Adversarial_Learning_of_Classifier_and_CVPR_2023_paper.pdf)), CVPR 2024 ([CTI](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf))

Beside, in ECCV 24, **"Diffusion-Guided Weakly Supervised Semantic Segmentation"** ([DiG](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06482.pdf)) is also accepted. Feel free to check our work! 

## 3.2 Acknowledgement
We heavily borrow the work from [MCTformer](https://github.com/xulianuwa/MCTformer), [PSA](https://github.com/jiwoon-ahn/psa) repository. Thanks for the excellent codes!

Also, we are greatly inspired by [What do neural networks learn in image classification? A frequency shortcut perspective](https://github.com/nis-research/nn-frequency-shortcuts). Thanks for the excellent work!
```
[1] Xu, et al. "Multi-class token transformer for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[2] Ahn, et al. "Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2018.
[3] Wang, et al. "What do neural networks learn in image classification? A frequency shortcut perspective." Proceedings of the IEEE/CVF international conference on computer vision. 2023.
