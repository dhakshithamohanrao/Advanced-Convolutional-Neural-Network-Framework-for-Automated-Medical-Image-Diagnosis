# Advanced Convolutional Neural Network Framework for Automated Medical Image Diagnosis

## Project Overview

This project develops a Convolutional Neural Network (CNN)-based framework for automated classification of chest X-ray images into three categories:

- **COVID-19**
- **Pneumonia**
- **Normal**

The project explores how deep learning can support medical image analysis by automatically learning visual patterns from chest X-ray images. A CNN model was implemented using PyTorch and evaluated through training and validation performance, test accuracy, classification metrics, confusion matrices, ROC curves, and precision–recall curves.

In addition, controlled experiments were conducted by changing individual hyperparameters and model configurations to examine their influence on classification performance.

**Important:** This project is an academic machine learning experiment and is not a clinically validated diagnostic system.

## Problem Statement

Timely identification of COVID-19 and pneumonia is important for appropriate patient care and disease management. However, healthcare providers may face challenges such as:

- High patient volumes
- Limited availability of specialist radiologists
- Delays in image interpretation
- Imbalanced medical imaging datasets

The objective is to develop an automated chest X-ray classification model that can distinguish between COVID-19, pneumonia, and normal cases. Such a system could provide preliminary decision-support information and help explore ways to improve screening efficiency.

## Objectives

1. Explore and understand the chest X-ray dataset and its class distribution.
2. Prepare images through resizing, grayscale conversion, normalization, and augmentation.
3. Address class imbalance using sampling and class-weighted loss.
4. Design and train a baseline CNN for three-class classification.
5. Evaluate model performance using accuracy, precision, recall, F1-score, confusion matrix, ROC curves, and precision–recall curves.
6. Conduct ten controlled experiments by modifying individual model or training parameters.
7. Compare experimental results and identify promising configurations.

## Dataset

**Dataset:** [Chest X-Ray COVID-19 & Pneumonia Dataset – Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

The dataset contains chest X-ray images organized into three categories.

| Class | Training Images | Test Images |
|---|---:|---:|
| COVID19 | 460 | 116 |
| NORMAL | 1,266 | 317 |
| PNEUMONIA | 3,418 | 855 |
| **Total** | **5,144** | **1,288** |

The dataset has an imbalanced class distribution, with pneumonia images representing the largest class and COVID-19 images the smallest.

## Technologies and Libraries

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)
- Kaggle Notebooks

## Methodology

### 1. Dataset Exploration

- Loaded the training and test image directories.
- Identified the three classification categories.
- Examined the number of images in each class.
- Visualized class distributions and sample chest X-ray images.

### 2. Image Preprocessing

The images were processed using the following transformations:

- Converted images to grayscale.
- Resized images to 224 × 224 pixels.
- Applied random horizontal flipping and rotation in the defined training transformation pipeline.
- Converted images into PyTorch tensors.
- Normalized pixel values using a mean of 0.5 and standard deviation of 0.5.

### 3. Dataset Splitting

- Used 80% of the training dataset for model training.
- Used the remaining 20% for validation.
- Kept the provided test dataset separate for final evaluation.

### 4. Class Imbalance Handling

Two techniques were implemented:

- **WeightedRandomSampler:** Used inverse class-frequency-based sample weights to balance the representation of classes during sampling.
- **Class-weighted CrossEntropyLoss:** Applied class weights during model training to account for differences in class frequencies.

### 5. CNN Architecture

A custom CNN model was developed using PyTorch.

| Layer | Configuration |
|---|---|
| Input | 1-channel grayscale image (224 × 224) |
| Convolution 1 | 32 filters, 3 × 3 kernel |
| Max Pooling | 2 × 2 |
| Convolution 2 | 64 filters, 3 × 3 kernel |
| Max Pooling | 2 × 2 |
| Convolution 3 | 128 filters, 3 × 3 kernel |
| Max Pooling | 2 × 2 |
| Fully Connected 1 | 256 neurons |
| Dropout | 0.3 |
| Output Layer | 3 classes |

ReLU activation was used in the convolutional and hidden fully connected layers. CrossEntropyLoss was used for multi-class classification.

### 6. Model Training

The baseline CNN was trained using the following configuration:

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Dropout | 0.3 |
| Loss Function | Weighted CrossEntropyLoss |
| Input Size | 224 × 224 |

The model with the highest validation accuracy was saved for subsequent evaluation.

### 7. Model Evaluation

The model was evaluated using:

- Training and validation loss curves
- Training and validation accuracy curves
- Accuracy heatmaps
- Test accuracy
- Classification report
- Confusion matrix
- Multi-class ROC curves
- Precision–recall curves
- Sample prediction visualizations

## Results

### Baseline Model Performance

The baseline CNN achieved a reported **test accuracy of 95.19%** on 1,288 test images.

| Metric | Result |
|---|---:|
| Test Accuracy | 95.19% |
| Macro Precision | 0.95 |
| Macro Recall | 0.94 |
| Macro F1-score | 0.94 |
| Weighted F1-score | 0.95 |

### Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| COVID19 | 0.98 | 0.91 | 0.95 | 116 |
| NORMAL | 0.89 | 0.95 | 0.92 | 317 |
| PNEUMONIA | 0.97 | 0.96 | 0.97 | 855 |

The model achieved particularly strong performance for pneumonia classification. COVID-19 classification achieved high precision, while its recall was comparatively lower at 0.91.

## Controlled Experiments

Ten experiments were recorded to investigate the effects of changing individual model and training configurations.

| Experiment | Modification | Validation Accuracy | Test Accuracy |
|---|---|---:|---:|
| Baseline | No change | 95.19%* | 95.19% |
| Exp 1 | Epochs: 10 → 15 | 95.10% | 95.00% |
| Exp 2 | Learning rate: 0.001 → 0.0005 | 94.50% | 94.40% |
| Exp 3 | Batch size: 32 → 64 | 94.80% | 94.60% |
| Exp 4 | Dropout: 0.3 → 0.5 | 94.20% | 94.30% |
| Exp 5 | ReLU → GELU | 95.50% | 94.40% |
| Exp 6 | Adam → SGD | 94.00% | 93.90% |
| Exp 7 | Basic → Stronger augmentation | 95.30% | 94.10% |
| Exp 8 | Convolutional layers: 3 → 4 | 95.70% | 94.60% |
| Exp 9 | Input size: 224 → 256 | 95.40% | 94.20% |
| Exp 10 | Default → He weight initialization | 95.60% | 94.50% |

*The experiment summary records 95.19% validation accuracy for the baseline, while the training log reports a best validation accuracy of 94.46%. This discrepancy should be checked against the original experiment outputs.*

### Key Observations

- The baseline model achieved the highest reported test accuracy of 95.19%.
- Experiment 8, which introduced an additional convolutional layer, achieved the highest recorded validation accuracy of 95.70%.
- GELU activation and He weight initialization also produced relatively high validation accuracy.
- The SGD optimizer experiment recorded the lowest test accuracy among the listed configurations.
- The highest validation accuracy did not necessarily correspond to the highest test accuracy.

These results illustrate how changes in architecture and training hyperparameters can influence model performance.

## Conclusion

This project implemented a CNN-based framework for classifying chest X-ray images into COVID-19, pneumonia, and normal categories.

The baseline model achieved a reported test accuracy of 95.19%, with strong class-wise precision, recall, and F1-scores. Class imbalance handling, model evaluation, and controlled experimentation formed important parts of the workflow.

The experiments demonstrated that different architectural and hyperparameter configurations produced different validation and test results. Overall, the project provided practical experience in deep learning, medical image classification, model evaluation, and experimental comparison using PyTorch.

## Limitations

- The dataset is sourced from a single public Kaggle repository and may not represent the diversity of real-world clinical images.
- The model uses chest X-ray images without additional patient information or clinical metadata.
- Only custom CNN architectures were explored.
- Explainability methods such as Grad-CAM were not implemented.
- The model was not clinically validated or deployed in a hospital environment.
- The training and validation subsets share the same underlying dataset object in the supplied implementation. Consequently, assigning the validation transform also changes the training transform; this should be corrected before interpreting the augmentation results.
- The reported baseline validation accuracy differs between the training log and experiment summary and requires verification.

## Future Improvements

- Explore transfer learning using ResNet, EfficientNet, and DenseNet.
- Implement explainable AI methods such as Grad-CAM.
- Evaluate the model using larger, multi-institutional datasets.
- Incorporate patient clinical information through multimodal learning.
- Develop a web-based or mobile interface for demonstration.
- Investigate model optimization for deployment on resource-constrained devices.
- Improve experiment reproducibility by using controlled random seeds and consistent data transformations.

## References

1. Kaggle – [Chest X-Ray COVID-19 & Pneumonia Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
3. World Health Organization – [Coronavirus Disease (COVID-19)](https://www.who.int/health-topics/coronavirus)
4. Litjens, G., et al. (2017). A Survey on Deep Learning in Medical Image Analysis. *Medical Image Analysis*.
5. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays. *arXiv*.

---

**Project Focus:** Deep Learning | Convolutional Neural Networks | Medical Image Classification | PyTorch | Model Evaluation | Hyperparameter Experimentation
