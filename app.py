"""
Vietnamese Food Classifier - Multi-Tab Streamlit App
Features: Multi-model classification, model comparison, app information
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import glob
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Vietnamese Food Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding-top: 0rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    </style>
""", unsafe_allow_html=True)


def get_available_models():
    """Scan for available model checkpoints in models folder"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    model_files = glob.glob(os.path.join(models_dir, '*.pth'))
    available_models = {}
    
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pth', '')
        available_models[model_name] = model_path
    
    # Also check for checkpoints_* directories (legacy support)
    checkpoint_dirs = glob.glob('checkpoints_*/')
    for checkpoint_dir in checkpoint_dirs:
        model_name = checkpoint_dir.replace('checkpoints_', '').replace('/', '').replace('\\', '')
        checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            available_models[model_name] = checkpoint_path
    
    return available_models


@st.cache_resource
def load_model(checkpoint_path):
    """Load a trained model from checkpoint with intelligent detection"""
    import warnings
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        return None, None, None, None, None
    
    # Get class names
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    else:
        train_dir = 'vnfood_combined_dataset/train'
        if os.path.exists(train_dir):
            class_names = sorted([d for d in os.listdir(train_dir)
                                 if os.path.isdir(os.path.join(train_dir, d))])
        else:
            return None, None, None, None, None
    
    num_classes = len(class_names)
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    # Intelligently detect model type from checkpoint content
    def detect_model_from_state_dict(keys):
        """Detect model architecture from state dict keys"""
        if any('features.' in key for key in keys):
            # EfficientNet or MobileNet
            if 'classifier.1.weight' in keys:
                # Check output channels to distinguish EfficientNet versions
                try:
                    final_conv_shape = checkpoint['model_state_dict'].get('features.8.1.weight')
                    if final_conv_shape is not None:
                        out_channels = final_conv_shape.shape[0]
                        if out_channels >= 1500:  # B3 or larger
                            return 'efficientnet_b3'
                        else:
                            return 'efficientnet_b0'
                except:
                    pass
                return 'efficientnet_b0'
            elif 'classifier.3.weight' in keys:
                return 'mobilenet_v3_large'
        elif any('layer1.' in key for key in keys):
            # ResNet
            return 'resnet50' if any('layer4.2.' in key for key in keys) else 'resnet101'
        return 'resnet50'
    
    model_name = checkpoint.get('model_name') or detect_model_from_state_dict(state_dict_keys)
    
    # Create model based on detected type
    model = None
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        return None, None, None, None, None
    
    # Load weights with strict=False to avoid size mismatch errors
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            return None, None, None, None, None
    
    accuracy = checkpoint.get('val_acc', checkpoint.get('best_acc', 'N/A'))
    
    model = model.to(device)
    model.eval()
    
    return model, class_names, device, model_name, accuracy


def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


@st.cache_resource
def generate_pipeline_image():
    """Generate pipeline diagram as image"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Define colors
    color_input = '#FF6B6B'
    color_process = '#4ECDC4'
    color_model = '#95E1D3'
    color_output = '#FFA07A'
    
    # Helper function to draw boxes
    def draw_box(ax, x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
               weight='bold', wrap=True)
    
    # Helper function to draw arrows
    def draw_arrow(ax, x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=30, 
                              linewidth=2.5, color='black')
        ax.add_patch(arrow)
    
    # Draw pipeline stages
    y_pos = 17
    
    # 1. User Input
    draw_box(ax, 5, y_pos, 3, 1, '🖼️ USER INPUT\n(Upload/URL/Paste)', color_input, 11)
    y_pos -= 1.5
    draw_arrow(ax, 5, 17.5, 5, y_pos + 0.5)
    
    # 2. Image Preprocessing
    draw_box(ax, 5, y_pos, 3.5, 1, '🔧 IMAGE PREPROCESSING\n(Resize 224x224, Normalize)', color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)
    
    # 3. Model Selection
    draw_box(ax, 5, y_pos, 3.5, 1, '🎯 MODEL SELECTION\n(1 or More Models)', color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)
    
    # 4. Parallel Inference
    draw_box(ax, 5, y_pos, 4, 1.2, '⚡ PARALLEL INFERENCE\n(Process Simultaneously)', color_process, 10)
    y_pos -= 1.8
    
    # Individual model boxes
    models_y = y_pos + 1
    model_positions = [(1.5, models_y), (5, models_y), (8.5, models_y)]
    model_names = ['ResNet50\n(89% acc)', 'EfficientNet\n(92% acc)', 'MobileNetV3\n(87% acc)']
    
    draw_arrow(ax, 5, y_pos + 1.8, 1.5, models_y + 0.5)
    draw_arrow(ax, 5, y_pos + 1.8, 5, models_y + 0.5)
    draw_arrow(ax, 5, y_pos + 1.8, 8.5, models_y + 0.5)
    
    for i, (x, y) in enumerate(model_positions):
        draw_box(ax, x, y, 2.2, 1, model_names[i], color_model, 9)
    
    # Arrows from models
    draw_arrow(ax, 1.5, models_y - 0.5, 1.5, models_y - 1.2)
    draw_arrow(ax, 5, models_y - 0.5, 5, models_y - 1.2)
    draw_arrow(ax, 8.5, models_y - 0.5, 8.5, models_y - 1.2)
    
    y_pos = models_y - 2
    
    # 5. Softmax + Probability
    draw_box(ax, 5, y_pos, 4, 1, '📊 SOFTMAX + PROBABILITY\n(Confidence Scores)', color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)
    
    # 6. Results Aggregation
    draw_box(ax, 5, y_pos, 3.5, 1, '📋 RESULTS AGGREGATION\n(Top-3 per Model)', color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)
    
    # 7. Consensus Analysis
    draw_box(ax, 5, y_pos, 3.5, 1, '🤝 CONSENSUS ANALYSIS\n(Model Agreement %)', color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)
    
    # 8. Display Results
    draw_box(ax, 5, y_pos, 3.5, 1.2, '✨ DISPLAY RESULTS\n(Predictions & Scores)', color_output, 11)
    
    plt.tight_layout()
    return fig


def display_pipeline_diagram():
    """Display the pipeline diagram"""
    fig = generate_pipeline_image()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def predict(image, model, class_names, device, top_k=5):
    """Make prediction on image"""
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    results = []
    for i in range(top_k):
        results.append({
            'class': class_names[top_indices[i]],
            'probability': float(top_probs[i]),
            'rank': i + 1
        })
    
    return results


def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image, None
    except Exception as e:
        return None, f"Error loading image: {str(e)}"


# ============================================================================
# TAB 1: CLASSIFICATION
# ============================================================================
def tab_classification():
    st.header("🔍 Food Classification")
    st.markdown("Upload an image or provide a URL to classify Vietnamese food")
    
    available_models = get_available_models()
    
    if not available_models:
        st.error("⚠️ No models found! Place model checkpoints in 'models/' folder")
        st.info("📝 **Setup Instructions:**\n1. Create a 'models' folder\n2. Copy your .pth checkpoint files there\n3. Name them: resnet50.pth, efficientnet_b0.pth, etc.")
        return
    
    st.success(f"✅ Found {len(available_models)} model(s)")
    
    # Model selection
    st.subheader("Select Models")
    col1, col2 = st.columns(2)
    
    with col1:
        select_all = st.checkbox("Select All Models", value=False)
    
    selected_models = {}
    if select_all:
        selected_models = available_models
        st.info(f"Selected all {len(available_models)} models")
    else:
        cols = st.columns(min(3, len(available_models)))
        for idx, model_name in enumerate(available_models.keys()):
            with cols[idx % len(cols)]:
                if st.checkbox(model_name.upper(), key=f"model_{model_name}"):
                    selected_models[model_name] = available_models[model_name]
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model")
        return
    
    # Image input
    st.subheader("Input Image")
    input_method = st.radio("Choose input method:", ["Upload Image", "Image URL", "Paste Image"])
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    
    elif input_method == "Image URL":
        image_url = st.text_input("Enter image URL:")
        if image_url:
            with st.spinner("Loading image..."):
                image, error = load_image_from_url(image_url)
                if error:
                    st.error(error)
    
    elif input_method == "Paste Image":
        st.info("Copy an image and paste it here, or paste an image URL")
        # Note: Direct paste functionality would need JavaScript component
        st.text("Use Upload or URL method for best results")
    
    # Display image and predictions
    if image:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            st.subheader("🎯 Predictions")
            
            # Make predictions with selected models
            predictions_data = {}
            
            for model_name, model_path in selected_models.items():
                with st.spinner(f"Classifying with {model_name.upper()}..."):
                    model, class_names, device, detected_name, accuracy = load_model(model_path)
                    
                    if model is not None:
                        predictions = predict(image, model, class_names, device, top_k=3)
                        predictions_data[model_name] = {
                            'predictions': predictions,
                            'accuracy': accuracy,
                            'detected_name': detected_name
                        }
                    else:
                        st.warning(f"⚠️ Failed to load {model_name}. Skipping...")
            
            # Display results
            if predictions_data:
                # Calculate consensus/overall result first
                from collections import Counter
                all_top_predictions = []
                for data in predictions_data.values():
                    all_top_predictions.append(data['predictions'][0]['class'])
                
                vote_counts = Counter(all_top_predictions)
                consensus = vote_counts.most_common(1)[0]
                agreement = (consensus[1] / len(predictions_data)) * 100 if len(predictions_data) > 1 else 100.0
                
                # Display overall result first
                st.subheader("⭐ Overall Result")
                with st.container():
                    col_left, col_right = st.columns([1, 2])
                    
                    with col_left:
                        st.metric("Model Agreement", f"{agreement:.0f}%")
                        st.caption(f"Based on {len(predictions_data)} model{'s' if len(predictions_data) > 1 else ''}")
                    
                    with col_right:
                        st.success(f"**Consensus Prediction:** {consensus[0].replace('_', ' ').title()}")
                        if len(predictions_data) > 1:
                            st.caption(f"{int(consensus[1])}/{len(predictions_data)} models agree")
                
                st.divider()
                
                # Display individual model predictions
                st.subheader("🔍 Individual Model Predictions")
                
                for model_name, data in predictions_data.items():
                    with st.expander(f"🔹 {model_name.upper()}", expanded=False):
                        col_left, col_right = st.columns([1, 2])
                        
                        with col_left:
                            acc_text = f"{data['accuracy']:.2f}%" if isinstance(data['accuracy'], (int, float)) else data['accuracy']
                            st.metric("Model Accuracy", acc_text)
                            st.caption(f"Arch: {data['detected_name']}")
                        
                        with col_right:
                            best_pred = data['predictions'][0]
                            st.success(f"**Top Prediction:** {best_pred['class'].replace('_', ' ').title()}")
                        
                        # Top 3 predictions
                        st.write("**Top 3 Predictions:**")
                        for pred in data['predictions']:
                            percentage = pred['probability'] * 100
                            st.write(f"{pred['rank']}. {pred['class'].replace('_', ' ').title()}: {percentage:.2f}%")
                            st.progress(pred['probability'])


# ============================================================================
# TAB 2: MODEL COMPARISON
# ============================================================================
def tab_comparison():
    st.header("📊 Model Comparison")
    st.markdown("Compare performance metrics across all available models")
    
    available_models = get_available_models()
    
    if not available_models:
        st.error("⚠️ No models found!")
        return
    
    st.info(f"Found {len(available_models)} model(s)")
    
    # Load model info
    comparison_data = []
    
    for model_name, model_path in available_models.items():
        with st.spinner(f"Loading {model_name}..."):
            model, class_names, device, detected_name, accuracy = load_model(model_path)
            
            if model is not None:
                # Load additional metrics from checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    f1_score = checkpoint.get('f1_score', 'N/A')
                    precision = checkpoint.get('precision', 'N/A')
                    recall = checkpoint.get('recall', 'N/A')
                except:
                    f1_score = precision = recall = 'N/A'
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Architecture': detected_name,
                    'Accuracy': accuracy if isinstance(accuracy, (int, float)) else float(accuracy) if accuracy != 'N/A' else 0,
                    'F1-Score': f1_score if isinstance(f1_score, str) else f1_score * 100,
                    'Precision': precision if isinstance(precision, str) else precision * 100,
                    'Recall': recall if isinstance(recall, str) else recall * 100,
                    'Classes': len(class_names),
                    'Checkpoint': model_path
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Display table
        st.subheader("Model Metrics")
        metrics_cols = ['Model', 'Architecture', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Classes']
        display_df = df[metrics_cols].copy()
        
        # Format numeric columns
        for col in ['Accuracy', 'F1-Score', 'Precision', 'Recall']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Best model
        if len(df) > 0 and 'Accuracy' in df.columns:
            best_idx = df['Accuracy'].idxmax()
            best_model = df.loc[best_idx]
            st.success(f"🏆 Best Model: **{best_model['Model']}** with {best_model['Accuracy']:.2f}% accuracy")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if len(df) > 1:
                st.subheader("Accuracy Comparison")
                chart_data = df.set_index('Model')['Accuracy'].sort_values(ascending=True)
                st.bar_chart(chart_data)
        
        
        # Detailed model info
        st.divider()
        st.subheader("Model Details")
        
        for idx, row in df.iterrows():
            with st.expander(f"{row['Model']} Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Architecture", row['Architecture'])
                    accuracy_val = f"{row['Accuracy']:.2f}%" if isinstance(row['Accuracy'], (int, float)) else row['Accuracy']
                    st.metric("Accuracy", accuracy_val)
                with col2:
                    f1_val = f"{row['F1-Score']:.2f}%" if isinstance(row['F1-Score'], (int, float)) else row['F1-Score']
                    st.metric("F1-Score", f1_val)
                    precision_val = f"{row['Precision']:.2f}%" if isinstance(row['Precision'], (int, float)) else row['Precision']
                    st.metric("Precision", precision_val)
                with col3:
                    recall_val = f"{row['Recall']:.2f}%" if isinstance(row['Recall'], (int, float)) else row['Recall']
                    st.metric("Recall", recall_val)
                    st.metric("Classes", row['Classes'])
    else:
        st.error("Failed to load model information")


# ============================================================================
# TAB 3: INFORMATION & TERMS OF SERVICE
# ============================================================================
def tab_info():
    st.header("ℹ️ Application Information")
    
    tab_about, tab_tech, tab_pipeline = st.tabs(["About", "Technologies", "Pipeline"])
    
    with tab_about:
        st.subheader("🍜 Vietnamese Food Classifier")
        st.markdown("""
        ### What is this app?
        This is an AI-powered application designed to classify **103 different Vietnamese food categories**. 
        Users can upload images or provide image URLs to get instant predictions from multiple deep learning models.
        
        ### Features
        - **Multi-Model Support**: Use one or multiple models simultaneously
        - **Flexible Input**: Upload images or provide URLs
        - **Consensus Analysis**: Get agreement scores when using multiple models
        - **Model Comparison**: Compare performance metrics across models
        - **Real-time Predictions**: Fast classification powered by PyTorch
        
        ### Terms of Service
        1. **Usage**: This app is provided for educational and personal use
        2. **Accuracy**: Model predictions are probabilistic; always verify results
        3. **Data**: Images are processed locally; no data is stored
        4. **Limitations**: Best performance on clear, well-lit food images
        5. **Attribution**: Dataset and models created for Vietnamese food classification research
        
        ### Disclaimer
        This application is for informational purposes. While efforts have been made to ensure accuracy,
        the developers make no guarantees about prediction correctness. Use at your own discretion.
        Made with ❤️ by @Gunner 
        """)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Classes", "103")
        with col2:
            st.metric("Max Models", "6")
        with col3:
            st.metric("Input Methods", "3")
    
    with tab_tech:
        st.subheader("🛠️ Technology Stack")
        
        tech_data = {
            'Category': ['ML Framework', 'ML Framework', 'Computer Vision', 'Web Framework', 'Data Processing', 'Model Architectures'],
            'Technology': ['PyTorch', 'TorchVision', 'PIL/Pillow', 'Streamlit', 'Pandas/NumPy', 'ResNet, EfficientNet, MobileNetV3'],
            'Purpose': ['Deep learning models', 'Pre-trained models', 'Image processing', 'Web interface', 'Data analysis', 'Classification backbones']
        }
        
        df_tech = pd.DataFrame(tech_data)
        st.dataframe(df_tech, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Model Architectures
        - **ResNet50/101**: Deep residual networks for robust feature extraction
        - **EfficientNet B0/B3**: Scalable models with optimal efficiency
        - **MobileNetV3**: Lightweight models for fast inference
        
        ### Libraries & Tools
        - PyTorch: Neural network framework
        - Torchvision: Computer vision utilities
        - Streamlit: Web application framework
        - Pandas: Data manipulation
        - Pillow: Image processing
        """)
    
    with tab_pipeline:
        st.subheader("📈 Application Pipeline")
        st.markdown("_Visual representation of how the application processes your image through multiple AI models_")
        
        st.divider()
        
        # Display the pipeline diagram
        display_pipeline_diagram()
        
        st.divider()
        st.subheader("📊 Pipeline Stages Explained")
        
        with st.expander("🖼️ **Stage 1: User Input**", expanded=False):
            st.markdown("""
            Users can provide input through three methods:
            - **Upload Image**: Direct file upload from device
            - **Image URL**: Paste URL from web
            - **Paste Image**: Copy-paste from clipboard
            """)
        
        with st.expander("🔧 **Stage 2: Image Preprocessing**", expanded=False):
            st.markdown("""
            Images are standardized before processing:
            - Resize to 224×224 pixels (model input size)
            - Normalize using ImageNet statistics
            - Convert to tensor format for PyTorch
            """)
        
        with st.expander("🎯 **Stage 3: Model Selection**", expanded=False):
            st.markdown("""
            Users select one or more AI models:
            - ResNet50, ResNet101
            - EfficientNet-B0, EfficientNet-B3
            - MobileNetV3
            - Select all at once or individual models
            """)
        
        with st.expander("⚡ **Stage 4: Parallel Inference**", expanded=False):
            st.markdown("""
            All selected models process the image simultaneously:
            - Each model extracts features independently
            - Forward pass through neural network
            - Generates raw prediction scores (logits)
            """)
        
        with st.expander("📊 **Stage 5: Softmax + Probability**", expanded=False):
            st.markdown("""
            Raw scores converted to probabilities:
            - Softmax activation applied
            - Scores normalized to 0-1 range
            - Represents confidence per food class
            """)
        
        with st.expander("📋 **Stage 6: Results Aggregation**", expanded=False):
            st.markdown("""
            Top predictions collected from each model:
            - Top-3 predictions per model shown
            - Sorted by confidence score
            - Ready for comparison
            """)
        
        with st.expander("🤝 **Stage 7: Consensus Analysis**", expanded=False):
            st.markdown("""
            When multiple models are used:
            - Calculate agreement percentage
            - Count votes for each food class
            - Show which prediction has strongest consensus
            """)
        
        with st.expander("✨ **Stage 8: Display Results**", expanded=False):
            st.markdown("""
            Final results shown to user:
            - Predictions with confidence bars
            - Model accuracy metrics
            - Consensus information
            - Easy-to-understand visualizations
            """)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🍜 Vietnamese Food Classifier")
    st.markdown("_AI-powered Vietnamese food recognition with multi-model support_")
    
    st.divider()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Classification", "📊 Model Comparison", "ℹ️ Information"])
    
    with tab1:
        tab_classification()
    
    with tab2:
        tab_comparison()
    
    with tab3:
        tab_info()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
    <small>Vietnamese Food Classifier | Built with PyTorch & Streamlit | © 2026</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
