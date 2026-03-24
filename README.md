## FusionGen

## :speech_balloon: Annoucement
- [2025.07.20] 🚩 **News:**  This paper is accepted by 2025 ICCV (DRL4Real) Oral🎉🎉🎉. The manuscript of FusionGen will be found in [FusionGen: Feature Fusion-Based Few-Shot EEG Data Generation](https://openaccess.thecvf.com/content/ICCV2025W/DRL4Real/html/Chen_FusionGen_Feature_Fusion-Based_Few-Shot_EEG_Data_Generation_ICCVW_2025_paper.html).
  
## 📌 Abstract
Brain-computer interfaces (BCIs) provide potential for applications ranging from medical rehabilitation to cognitive state assessment by establishing direct communication pathways between the brain and external devices via electroencephalography (EEG). However, EEG-based BCIs are severely constrained by data scarcity and significant inter-subject variability, which hinder the generalization and applicability of EEG decoding models in practical settings. To address these challenges, we propose FusionGen, a novel EEG data generation framework based on disentangled representation learning and feature fusion. By integrating features across trials through a feature matching fusion module and combining them with a lightweight feature extraction and reconstruction pipeline, FusionGen ensures both data diversity and trainability under limited data constraints. Extensive experiments on multiple publicly available EEG datasets demonstrate that FusionGen significantly outperforms existing augmentation techniques, yielding notable improvements in classification accuracy.

![FusionGen](./FusionGen/pic/FusionGen.png)

![Gen_visual](https://github.com/Hulianye/FusionGen/blob/main/Gen_Visual.png)

## 🚀  Contributions
- 🧩 We propose FusionGen, a few-shot EEG data generation framework that enhances generalization and scalability in brain–computer interface applications.
- 🛠️ We introduce a feature matching fusion module that integrates cross-sample features in the latent representation space and reconstructs high-fidelity EEG signals from these fused embeddings. 
- 📊 We validate FusionGen on multiple EEG datasets on MI and SSVEP paradigms, showing consistent accuracy improvements in few-shot scenarios.

## Repository Structure
- `FusionGen_SingleSubject.py`: FusionGen evaluation for single-subject experiments
- `FusionGen_CrossSubject.py`: FusionGen evaluation for cross-subject experiments
- `data_augment_SingleSubject.py`: baseline augmentation evaluation (single-subject)
- `data_augment_CrossSubject.py`: baseline augmentation evaluation (cross-subject)
- `FusionGen_BaseProcessor.py`: shared training/evaluation utilities
- `model/model.py`: model definitions (including FusionGen modules)
- `data/`: dataset loading and preprocessing scripts
- `requirements.txt`: Python dependencies

## Installation

Install all dependencies with:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

After installation, run any experiment script directly from the project root:

```bash
python data_augment_SingleSubject.py
python data_augment_CrossSubject.py
python FusionGen_SingleSubject.py
python FusionGen_CrossSubject.py
```

## Data Preparation
No manual dataset conversion is required for the default workflow.  
Datasets are loaded via the scripts in `data/`:
- `data/dataset001.py`
- `data/dataset002.py`
- `data/datasetZhou2016.py`

Depending on your selected script and dataset, required EEG data may be downloaded automatically on first run through dataset APIs (for example, MOABB/MNE-related pipelines). The first run can take longer due to downloading/caching.

## Run Experiments

### Baseline augmentation (single-subject)
```bash
python data_augment_SingleSubject.py
```

### Baseline augmentation (cross-subject)
```bash
python data_augment_CrossSubject.py
```

### FusionGen (single-subject)
```bash
python FusionGen_SingleSubject.py
```

### FusionGen (cross-subject)
```bash
python FusionGen_CrossSubject.py
```

## Parameter Customization
Default experiment parameters are defined in each script's `__main__` section.  
You can directly edit values such as:
- `subject_num`
- `train_num`
- `seed_range`
- `methods` (for augmentation baseline scripts)

Common edit points:
- `FusionGen_SingleSubject.py`
- `FusionGen_CrossSubject.py`
- `data_augment_SingleSubject.py`
- `data_augment_CrossSubject.py`

## Citation
If you find this project useful, please cite the FusionGen paper:

```bibtex
@inproceedings{chen2025fusiongen,
  title={FusionGen: Feature Fusion-Based Few-Shot EEG Data Generation},
  author={Chen, Yuheng and Liu, Dingkun and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year={2025}
}
```

## 📩 Contact
For any questions or collaborations, please feel free to reach out via `chenyuheng@hust.edu.cn` / `liudingkun@hust.edu.cn` or open an issue in this repository.
