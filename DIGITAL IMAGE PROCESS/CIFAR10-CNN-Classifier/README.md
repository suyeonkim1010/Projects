# 🧠 CIFAR10 CNN Classifier

This is a custom-built Convolutional Neural Network (CNN) trained from scratch (no pretrained models) to classify images from the CIFAR-10 dataset using PyTorch.

The model achieves **85.4%** accuracy on the test set and meets all constraints such as running on a Colab GPU with >200 images/sec and checkpoint size <400MB.

---

## 📁 Project Structure

```
CIFAR10-CNN-Classifier/
├── checkpoints/            # Best model weights
│   └── model.pt.114
├── model.py                # CNN architecture definition
├── train.py                # Training pipeline
├── utils.py                # Utility functions (e.g., data loading)
├── training_run.ipynb      # Evaluation script (based on notebook logic)
└── requirements.txt        # Required packages
```

---

## 🧪 Performance

| Metric         | Value        |
|----------------|--------------|
| Test Accuracy  | **85.4%**    |
| Model Size     | ~6.5MB       |
| Training Speed | >9,000 img/s |

---

## 🚀 How to Run

1. **Install requirements**
```bash
pip install -r requirements.txt
```

2. **Train the model**
```bash
python train.py
```

3. **Evaluate the saved checkpoint**
```bash
python training_run.py
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- numpy
- tqdm

---

## 📦 Dataset

CIFAR-10 (Automatically downloaded via `torchvision.datasets.CIFAR10`)

---

## 🐝 Author
Developed by **[Suyeon Kim]**. Feel free to reach out if you have any questions or suggestions!  
GitHub Profile(https://github.com/suyeonkim1010/Projects.git)  
LinkedIn Profile(https://www.linkedin.com/in/suyeon-kim-a43730256/) 
