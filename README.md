# 🍜 Vietnamese Food Classifier

A deep learning application that classifies images of Vietnamese dishes into 103 different food categories using PyTorch and Streamlit.

## 📋 Features

- **103 Vietnamese Food Categories** - Recognizes a wide variety of traditional Vietnamese dishes
- **Multiple Model Architectures** - Supports ResNet50/101, EfficientNet B0/B3, and MobileNet V3
- **Interactive Web Interface** - Built with Streamlit for easy image classification
- **Flexible Input Methods** - Upload images or provide image URLs
- **Auto Model Detection** - Automatically detects the trained model architecture from checkpoints
- **Transfer Learning** - Uses ImageNet pre-trained weights for better performance

## Application
a small deployment is located here https://food-deploy-test-epenuknzfaey2iccuwrdx4.streamlit.app , which contains eff_b0 models, the others are to large to be included

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a model with default settings (EfficientNet-B0):

```bash
python train.py
```

Train with custom settings:

```bash
python train.py --model efficientnet_b0 --epochs 30 --batch_size 32 --lr 0.001
```

Available models:
- `resnet50`
- `resnet101`
- `efficientnet_b0` (default)
- `efficientnet_b3`
- `mobilenet_v3_large`

### Run the Web App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📊 Dataset

This project uses the [VNFood Dataset](https://www.kaggle.com/datasets/meowluvmatcha/vnfood-30-100) containing 103 Vietnamese food categories.

Dataset structure:
```
vnfood_combined_dataset/
├── train/
├── val/
└── test/
```

## 🎯 Model Performance

The model achieves high accuracy on Vietnamese food classification. Training logs and checkpoints are saved in the `checkpoints/` directory.

## 🛠️ Project Structure

```
vietnamese-food-training/
├── app.py                      # Streamlit web application
├── train.py                    # Training script
├── requirements.txt            # Python dependencies
├── checkpoints/               # Saved model checkpoints
│   └── best_checkpoint.pth    # Best performing model
└── vnfood_combined_dataset/   # Dataset directory
    ├── train/
    ├── val/
    └── test/
```

## 💡 Usage Examples

### Using the Web App

1. **Upload an image** - Click "Browse files" to upload a food image
2. **Provide a URL** - Paste an image URL in the text field
3. **View predictions** - See top predictions with confidence scores

### Using the Trained Model

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
checkpoint = torch.load('checkpoints/best_checkpoint.pth')
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 103)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('food_image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(1).item()
```

## 📝 Training Configuration

Default configuration in `train.py`:

- **Batch size**: 32
- **Learning rate**: 0.001
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 30
- **Data augmentation**: Random crop, flip, rotation, color jitter

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [VNFood Dataset on Kaggle](https://www.kaggle.com/datasets/meowluvmatcha/vnfood-30-100)
- Pre-trained models: PyTorch torchvision models
- Framework: PyTorch, Streamlit

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
