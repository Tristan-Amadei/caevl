# [WACV 2026] Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone

**Official repository of the paper**:  
[*"Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone"*](https://arxiv.org/abs/2512.02737) <br>
Tristan Amadei, Enric Meinhardt-Llopis, Benedicte Bascle, Corentin Abgrall, Gabriele Facciolo  

---

## 🧠 Overview

We introduce **CAEVL**, a data-efficient method for UAV geo-localization that does not require paired UAV–satellite images during training.

Unlike traditional cross-view localization approaches that rely on aligned image pairs, CAEVL learns a shared representation space using:

- edge-based image representations  
- self-supervised learning  
- non-contrastive training objectives  

This allows the model to generalize to challenging conditions such as:
- low-quality UAV imagery  
- high-altitude viewpoints  
- strong appearance differences between modalities  

---

## ⚙️ Method Summary

CAEVL is based on a **cross-view embedding learning strategy**:

- UAV and satellite images are encoded into a shared latent space  
- Training is performed **without explicit UAV–satellite correspondences**  
- A non-contrastive objective structures the latent space  
- Edge representations improve robustness to appearance variations  

At inference time:
- a UAV image is encoded  
- nearest neighbors are retrieved from a satellite database  
- the predicted location is obtained via image matching

---
## Results

CAEVL is a very lightweight method with a computational cost of only 1.4 GFLOPs, and is competitive compared to SOTA methods despite having been trained with reference satellite imagery only.

| Method | Data | R@1 (100m) | R@1 (250m) | GFLOPs |
|---|---|---|---|---|
| Mix VPR | Paired | 42.2 | 79.2 | <u>10.3</u> |
| Eigen Places | Paired | 39.8 | 78.7 | 19.7 |
| FSRA | Paired | 37.2 | <u>84.7</u> | 13.3 |
| DAC | Paired | <u>48.6</u> | **85.5** | 20.6 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| Di Piazza et al. | Sat-Only | 26.3 | 34.1 | **1.4** |
| **CAEVL (Ours)** | **Sat-Only** | **49.7** | 82.8 | **1.4** |

---

## 📦 ViLD Dataset

We release the **ViLD dataset**, designed for UAV-to-satellite matching and geo-localization tasks.

### Contents
- UAV and satellite imagery from multiple flights  
- Ground-truth coordinates (Lambert 93 and UTM)  
- Predefined train/validation/test splits  
- Tools for visualization and reproducibility  

👉 **Download on Zenodo (https://zenodo.org/records/19223815)** <br>
or <br>
👉 **Download from the [ENS website](https://kiwi.cmla.ens-cachan.fr/index.php/s/PkB9FgzETpA5iMR)**

---

## 🔐 Dataset Access

The dataset is distributed as a password-protected archive.

To request access, please send an email to:  
📧 vild.dataset@gmail.com  

Please include:
- Name and affiliation  
- Intended use (e.g., research, commercial, personal project)  

### 📄 Email template

Subject: ViLD Dataset Access Request

Hello,

I would like to request access to the ViLD dataset.

Name: [Your Name] <br>
Affiliation: [Your Institution / Company] <br>
Intended use: [Brief description]

Thank you


You will receive the password upon request.

---

## 🚀 Getting Started

### Evaluation
Here is an example of how to evaluate the model
```python
python caevl/evaluation/eval.py \
    --method=caevl \
    --weights=caevl/models/trained/stagetwo/stagetwo.pth \
    --database_folder=path/to/database \
    --queries_folder=path/to/queries \
    --database_coords_path=path/to/database_coords \
    --queries_coords_path=path/to/queries_coords \
```

This will create a log file in logs/log_dir. You can add --save_predictions to save the predictions, allowing you to visualize and analyze them afterwards.<br>
The --database_coords_path and --queries_coords_path parameters are paths to the dictionaries that contain the coordinates of the database and query images. They need to have the names of images as keys and the coordinates as values. Otherwise, the coordinates of the images can be stored directly in the filenames, as such @utm_east@utm_north@filename

### Training
Training the autoencoder for stage one is straightforward
```python
python caevl/ae/train.py --config=caevl/ae/configs/config_ae.yml
```

Fine-tuning the encoder during stage 2 is performed similarly:
```python
python caevl/ft_stage/train.py --config=caevl/ft_stage/configs/config_stagetwo.yml
```

You just need to indicate in the config file the folder in which the weights of the encoder are located, for instance 
```
architecture:
  dir_model: 'AutoEncoder'
  backbone: 'stageone'
```

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{amadei2026beyond,
  title={Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone},
  author={Amadei, Tristan and Meinhardt-Llopis, Enric and Bascle, Benedicte and Abgrall, Corentin and Facciolo, Gabriele},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7409--7419},
  year={2026}
}
```

## 📬 Contact

For questions related to the dataset or the method, feel free to contact us via the dataset email:
vild.dataset@gmail.com

## Acknowledgements
We thank the contributors of the open source codes including Dinov2 (https://github.com/facebookresearch/dinov2) and (https://github.com/gmberton/VPR-methods-evaluation).
