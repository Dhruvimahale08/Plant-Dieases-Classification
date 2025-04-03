# 🌿 Plant Disease Classification System

## 📌 Overview
This project implements a deep learning-based system to classify plant diseases using the PlantVillage dataset. We evaluated five state-of-the-art architectures (Vision Transformer, Xception, ResNet50, EfficientNetV2, and MobileNetV3) to identify the most accurate and efficient model for real-world deployment. Our Vision Transformer (ViT) implementation achieved **97% validation accuracy**, demonstrating superior performance for this classification task.

## 📂 Dataset
**PlantVillage Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset))
- **38 classes** of healthy and diseased plant leaves
- **54,305 images** total (train/validation split)
- **Classes include**: 
  - Tomato diseases (Early Blight, Late Blight, Leaf Mold)
  - Corn diseases (Common Rust, Gray Leaf Spot)
  - Potato diseases (Early Blight, Late Blight)
  - And many more (Apple, Cherry, Grape, etc.)
- **Image resolution**: 256x256px (resized to model-specific inputs)

## 🏗️ Model Architectures

### 1. Vision Transformer (ViT) - **97% Accuracy**
- **Architecture**:
  - Patch-based transformer encoder
  - 12 attention heads
  - 768-dimensional embeddings
  - Custom classification head
- **Strengths**: Best accuracy, excels with global feature relationships
- **Compute**: GPU-intensive (~15GB VRAM required)

### 2. Xception - **89.5% Accuracy**
- **Architecture**:
  - Depthwise separable convolutions
  - 71-layer deep
  - Modified FC layer for 38-class output
- **Strengths**: Excellent accuracy with reasonable compute

### 3. EfficientNetB7 - **73.8% Accuracy**
- **Architecture**:
  - Compound scaling (width/depth/resolution)
  - MBConv blocks with Fused-MBConv
  - B0 variant used
- **Strengths**: Good speed-accuracy tradeoff

### 4. ResNet50 - **92.1% Accuracy**
- **Architecture**:
  - 50-layer residual network
  - Global average pooling
  - Custom dense layers
- **Strengths**: Reliable baseline performance

### 5. MobileNetV3 - **90.3% Accuracy**
- **Architecture**:
  - Lightweight CNN with squeeze-excitation
  - Hard-swish activations
  - Small variant used
- **Strengths**: Fastest inference (ideal for mobile)

## 🚀 Key Features

1. **Comprehensive Evaluation**:
   - Direct comparison of 5 modern architectures
   - Detailed accuracy/latency metrics
   - Resource requirements analysis

2. **Optimized Training**:
   - Advanced data augmentation
   - Class imbalance handling
   - Learning rate scheduling

3. **Reproducible Results**:
   - Complete training logs
   - Saved model weights
   - Jupyter notebooks for each architecture

4. **Deployment Ready**:
   - TensorFlow SavedModel format
   - ONNX export support
   - Flask API example

## 📊 Performance Comparison

| Model          | Accuracy | Parameters | Inference Time (ms) | VRAM Usage |
|----------------|----------|------------|---------------------|------------|
| ViT            | 97.0%    | 86M        | 42                  | 15GB       |
| Xception       | 89.5%    | 22M        | 28                  | 8GB        |
| EfficientNetB7 | 73.8%    | 7M         | 18                  | 6GB        |
| ResNet50       | 92.1%    | 25M        | 22                  | 7GB        |
| MobileNetV3    | 90.3%    | 2M         | 9                   | 3GB        |



## 🛠 Kaggle-Specific Setup
```python
# In Kaggle Notebook:
from kaggle_datasets import KaggleDatasets
GCS_PATH = KaggleDatasets().get_gcs_path('plantvillage-dataset')

# Configure TPU/GPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
```

## 📈 Training Curves
![Training Progress]("C:\Users\Dhruvi\Downloads\training_history.png")
*Fig 2. ViT training/validation metrics*

## 🗂 Dataset Structure (Kaggle)
```
/plantvillage-dataset/
├── train/
│   ├── Tomato___Early_blight/
│   ├── Corn___Common_rust/
│   └── ... (38 classes)
└── validation/
    ├── Tomato___Late_blight/
    └── ... (same structure)
```

## 💡 Key Insights
1. ViT outperformed CNNs but required 3x more training time
2. MobileNetV3 provided best speed-accuracy tradeoff
3. Data augmentation boosted accuracy by ~8% across models

## 🌟 Best Practices (Kaggle)
```python
# Enable mixed-precision
policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy(policy)

# Optimize dataset pipeline
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a PR for:
- New model architectures
- Performance optimizations
- Additional datasets


## ✉️ Contact
For questions or collaborations:
- Dhruvi Mahale
[22cs036@charusat.edu.in]
- Shruti Panchal
[22cs044@charusat.edu.in] 
- [[LinkedIn Profile](https://www.linkedin.com/in/dhruvi-mahale-4aa072258/)]  

---
