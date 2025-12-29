"""
Streamlit App for Vietnamese Food Classification
Classifies images of Vietnamese dishes into 103 different food categories
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import base64
from io import BytesIO
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Vietnamese Food Classifier",
    page_icon="🍜",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get class names from dataset directory
    train_dir = 'vnfood_combined_dataset/train'
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    
    # Load checkpoint first to detect model architecture
    checkpoint_path = 'checkpoints/best_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        st.error("⚠️ Checkpoint not found. Please train the model first.")
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to detect model type from checkpoint or state_dict structure
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        # Infer from state_dict keys
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        if any('features.' in key for key in state_dict_keys):
            # EfficientNet or MobileNet structure
            if 'classifier.1.weight' in state_dict_keys:
                # Count number of feature blocks to distinguish efficientnet versions
                max_feature_num = max([int(key.split('.')[1]) for key in state_dict_keys if key.startswith('features.') and key.split('.')[1].isdigit()])
                if max_feature_num == 8:
                    model_name = 'efficientnet_b0'
                elif max_feature_num > 8:
                    model_name = 'efficientnet_b3'
                else:
                    model_name = 'efficientnet_b0'
            elif 'classifier.3.weight' in state_dict_keys:
                model_name = 'mobilenet_v3_large'
            else:
                model_name = 'efficientnet_b0'
        elif any('layer1.' in key for key in state_dict_keys):
            # ResNet structure
            if any('layer4.2.' in key for key in state_dict_keys):
                model_name = 'resnet50'
            else:
                model_name = 'resnet101'
        else:
            model_name = 'resnet50'  # Default fallback
        
        st.info(f"🔍 Auto-detected model type: {model_name}")
    
    # Create model architecture
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
    else:
        st.error(f"⚠️ Unsupported model type: {model_name}")
        return None, None, None
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('val_acc', checkpoint.get('best_acc', 'N/A'))
        st.success(f"✅ Model loaded! Type: {model_name}, Accuracy: {accuracy:.2f}%" if isinstance(accuracy, (int, float)) else f"✅ Model loaded! Type: {model_name}")
    except Exception as e:
        st.error(f"⚠️ Error loading model weights: {str(e)}")
        st.info(f"💡 The checkpoint might be from a different model architecture. Detected: {model_name}")
        return None, None, None
    
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict(image, model, class_names, device, top_k=5):
    """Make prediction on the image"""
    transform = get_transform()
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top K predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        results.append({
            'class': class_names[idx],
            'probability': float(prob),
            'percentage': float(prob * 100)
        })
    
    return results

def format_dish_name(dish_name):
    """Format dish name for display (replace hyphens with spaces and capitalize)"""
    return dish_name.replace('-', ' ').title()

def paste_image_component():
    """Create a paste image component using JavaScript"""
    paste_html = """
    <div style="border: 2px dashed #4CAF50; padding: 20px; border-radius: 10px; text-align: center; background-color: #f0f8f0; margin: 10px 0;">
        <h4 style="color: #4CAF50; margin-top: 0;">📋 Paste Image or URL Here (Ctrl+V)</h4>
        <p style="color: #666;">Click inside this box and press <strong>Ctrl+V</strong> to paste an image or image URL from your clipboard</p>
        <canvas id="pasteCanvas" style="display:none;"></canvas>
        <div id="pasteStatus" style="margin-top: 10px; color: #666;"></div>
    </div>
    <script>
        const pasteArea = document.querySelector('div');
        const canvas = document.getElementById('pasteCanvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('pasteStatus');
        
        // Make the div focusable
        pasteArea.setAttribute('tabindex', '0');
        
        function loadImageFromUrl(url) {
            status.innerHTML = '<span style="color: #2196F3;">⏳ Loading image from URL...</span>';
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // Convert canvas to base64
                const base64data = canvas.toDataURL('image/png');
                
                // Send to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    data: base64data
                }, '*');
                
                status.innerHTML = '<span style="color: #4CAF50;">✓ Image loaded from URL successfully!</span>';
            };
            
            img.onerror = function() {
                status.innerHTML = '<span style="color: #f44336;">✗ Failed to load image from URL. Make sure it\'s a valid image URL.</span>';
            };
            
            img.src = url;
        }
        
        pasteArea.addEventListener('paste', async (e) => {
            e.preventDefault();
            const items = e.clipboardData.items;
            let foundImage = false;
            
            // First, try to find an image
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    foundImage = true;
                    const blob = item.getAsFile();
                    const reader = new FileReader();
                    
                    reader.onload = function(event) {
                        const img = new Image();
                        img.onload = function() {
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx.drawImage(img, 0, 0);
                            
                            // Convert canvas to base64
                            const base64data = canvas.toDataURL('image/png');
                            
                            // Send to Streamlit
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                data: base64data
                            }, '*');
                            
                            status.innerHTML = '<span style="color: #4CAF50;">✓ Image pasted successfully!</span>';
                        };
                        img.src = event.target.result;
                    };
                    reader.readAsDataURL(blob);
                    break;
                }
            }
            
            // If no image found, check for text (URL)
            if (!foundImage) {
                for (let item of items) {
                    if (item.type === 'text/plain') {
                        item.getAsString(function(text) {
                            text = text.trim();
                            // Check if it looks like a URL
                            if (text.match(/^https?:\/\/.+\.(jpg|jpeg|png|gif|webp|bmp)/i) || 
                                text.match(/^https?:\/\/.+/i)) {
                                loadImageFromUrl(text);
                            } else {
                                status.innerHTML = '<span style="color: #ff9800;">⚠ Please paste an image or a valid image URL</span>';
                            }
                        });
                        break;
                    }
                }
            }
        });
        
        // Focus hint
        pasteArea.addEventListener('click', () => {
            pasteArea.focus();
            status.innerHTML = '<span style="color: #2196F3;">Ready to paste! Press Ctrl+V</span>';
        });
        
        pasteArea.addEventListener('blur', () => {
            status.innerHTML = '';
        });
    </script>
    """
    
    pasted_data = components.html(paste_html, height=180)
    return pasted_data

# Main App
def main():
    # Title and description
    st.title("🍜 Vietnamese Food Classifier")
    st.markdown("""
    Upload an image of a Vietnamese dish and the AI will identify it!  
    This model can recognize **103 different Vietnamese food categories**.
    """)
    
    # Load model
    with st.spinner('Loading model...'):
        model, class_names, device = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of predictions to show", 1, 10, 5)
        
        st.header("📊 Model Info")
        st.info(f"""
        - **Classes**: {len(class_names)}
        - **Device**: {device}
        """)
        
        st.header("📚 About")
        st.markdown("""
        This classifier uses deep learning to identify Vietnamese dishes.
        The model was trained on the VNFood dataset containing 103 categories.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of Vietnamese food"
    )
    
    # OR use URL input
    st.markdown("### 🔗 Or enter an image URL")
    image_url = st.text_input("Paste image URL here", placeholder="https://example.com/image.jpg")
    
    # Process images
    image_to_process = None
    image_source = None
    
    # Check URL first
    if image_url and image_url.strip():
        try:
            import requests
            from io import BytesIO
            
            with st.spinner('Loading image from URL...'):
                response = requests.get(image_url.strip(), timeout=10)
                response.raise_for_status()
                image_to_process = Image.open(BytesIO(response.content)).convert('RGB')
                image_source = "url"
                st.success("✅ Image loaded from URL successfully!")
        except Exception as e:
            st.error(f"❌ Error loading image from URL: {str(e)}")
            image_to_process = None
    
    # Check uploaded file
    if uploaded_file is not None and image_to_process is None:
        image_to_process = Image.open(uploaded_file).convert('RGB')
        image_source = "uploaded"
    
    st.markdown("---")
    
    # Process uploaded image
    if image_to_process is not None:
        # Display image
        image = image_to_process
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📸 {'URL' if image_source == 'url' else 'Uploaded'} Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Predictions")
            
            with st.spinner('Analyzing image...'):
                predictions = predict(image, model, class_names, device, top_k)
            
            # Display top prediction prominently
            top_pred = predictions[0]
            st.success(f"**{format_dish_name(top_pred['class'])}**")
            st.metric("Confidence", f"{top_pred['percentage']:.2f}%")
            
            # Display all predictions
            st.markdown("### Top Predictions:")
            for i, pred in enumerate(predictions):
                with st.container():
                    st.markdown(f"**{i+1}. {format_dish_name(pred['class'])}**")
                    st.progress(pred['probability'])
                    st.caption(f"Confidence: {pred['percentage']:.2f}%")
    
    else:
        # Show instructions when no image is uploaded
        st.info("👆 Upload an image to get started!")
        
        # Show all available categories in an expander
        with st.expander("📋 View all 103 food categories"):
            cols = st.columns(3)
            for idx, cls in enumerate(class_names):
                with cols[idx % 3]:
                    st.markdown(f"- {format_dish_name(cls)}")

if __name__ == '__main__':
    main()
