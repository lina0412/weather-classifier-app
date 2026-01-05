# streamlit_app.py - WITH CONFIDENCE THRESHOLD
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time
import pandas as pd
import plotly.graph_objects as go
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Weather Classifier Pro",
    page_icon="⛈️",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .reject-box {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        color: #721C24;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.markdown('<h1 class="main-title">⛈️ Weather Classifier PRO</h1>', unsafe_allow_html=True)
st.markdown("### Smart classification with confidence thresholding")

# ==================== SETTINGS ====================
CONFIDENCE_THRESHOLD = 0.7  # Minimum 70% confidence to accept prediction
MAX_IMAGE_SIZE = 5000  # Max image dimension in pixels

# ==================== LOAD DATA ====================
@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

@st.cache_data  
def load_accuracies():
    return {
        "Sparse Fine-Tuning": 0.9315,
        "Fine-Tuning (Last 30 layers)": 0.9259,
        "Feature Extraction + SVM": 0.7963,
        "Stochastic Fine-Tuning": 0.7593,
        "Feature Extraction + Random Forest": 0.7778,
        "Feature Extraction + MLP": 0.7222,
        "Knowledge Distillation": 0.7056,
    }

# ==================== IMAGE VALIDATION ====================
def validate_image(image):
    """Check if image is likely a weather photo"""
    issues = []
    
    # Check image size
    if max(image.size) > MAX_IMAGE_SIZE:
        issues.append(f"Image too large: {image.size}")
    
    # Check if image is mostly gray (low color variation)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        color_std = np.std(img_array, axis=(0, 1))
        if np.mean(color_std) < 20:  # Low color variation
            issues.append("Image appears grayscale/low color")
    
    return issues

# ==================== MODEL LOADING ====================
@st.cache_resource
def create_feature_extractor():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

@st.cache_resource
def load_models():
    """Load all available models"""
    models = {}
    
    # Load Keras models
    keras_models = {
        "Sparse Fine-Tuning": "strategy4_sparse.keras",
        "Fine-Tuning (Last 30 layers)": "strategy2_fine_tuned.keras",
        "Stochastic Fine-Tuning": "strategy3_stochastic.keras",
        "Knowledge Distillation": "weather_classifier_fixed.keras"
    }
    
    for name, file in keras_models.items():
        if os.path.exists(file):
            try:
                models[name] = {
                    'type': 'keras',
                    'model': tf.keras.models.load_model(file),
                    'file': file
                }
            except:
                pass
    
    # Load ML models
    ml_models = {
        "Feature Extraction + SVM": "strategy1_svm_rbf.pkl",
        "Feature Extraction + Random Forest": "strategy1_random_forest.pkl",
        "Feature Extraction + MLP": "strategy1_mlp.pkl"
    }
    
    for name, file in ml_models.items():
        if os.path.exists(file):
            try:
                models[name] = {
                    'type': 'ml',
                    'model': joblib.load(file),
                    'file': file
                }
            except:
                pass
    
    return models

# ==================== PREDICTION WITH REJECTION ====================
def predict_with_rejection(model_info, image_array, feature_extractor=None):
    """Make prediction with confidence check"""
    start_time = time.time()
    
    if model_info['type'] == 'keras':
        predictions = model_info['model'].predict(image_array, verbose=0)[0]
    else:  # ML model
        features = feature_extractor.predict(image_array, verbose=0)
        features_flat = features.reshape(features.shape[0], -1)
        
        if hasattr(model_info['model'], 'predict_proba'):
            predictions = model_info['model'].predict_proba(features_flat)[0]
        else:
            pred_class = model_info['model'].predict(features_flat)[0]
            predictions = np.zeros(5)
            predictions[pred_class] = 1.0
    
    prediction_time = time.time() - start_time
    
    # Get max confidence
    max_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    # Check if confidence is too low
    if max_confidence < CONFIDENCE_THRESHOLD:
        return None, prediction_time, max_confidence  # Reject prediction
    
    return predictions, prediction_time, max_confidence

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Confidence threshold slider
    st.subheader("🔐 Confidence Settings")
    confidence_threshold = st.slider(
        "Minimum confidence required:",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Higher = more strict, Lower = more lenient"
    )
    
    # Model selection
    st.subheader("🤖 Model Selection")
    model_options = {
        "Sparse Fine-Tuning": "strategy4_sparse.keras",
        "Fine-Tuning": "strategy2_fine_tuned.keras",
        "Stochastic": "strategy3_stochastic.keras",
        "Knowledge Distillation": "weather_classifier_fixed.keras",
        "SVM": "strategy1_svm_rbf.pkl",
        "Random Forest": "strategy1_random_forest.pkl",
        "MLP": "strategy1_mlp.pkl"
    }
    
    # Check which models exist
    available_models = []
    for name, file in model_options.items():
        if os.path.exists(file):
            available_models.append(name)
    
    selected_names = st.multiselect(
        "Select strategies to compare:",
        available_models,
        default=["Sparse Fine-Tuning", "Knowledge Distillation"] if "Sparse Fine-Tuning" in available_models else available_models[:2]
    )
    
    # Image upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a weather image:",
        type=['jpg', 'jpeg', 'png'],
        help="Upload images of: hail, lightning, rain, sandstorm, snow"
    )
    
    # Example images button
    if st.button("ℹ️ Show Example Images"):
        st.info("""
        **Good examples:**
        - ⛈️ Storm clouds
        - 🌧️ Rain falling
        - ❄️ Snow landscape
        - ⚡ Lightning strike
        - 🌪️ Dust storm
        
        **Will be rejected:**
        - 👨‍💻 People/faces
        - 🏠 Buildings
        - 🐱 Animals
        - 📱 Text/screens
        - 🍎 Food/objects
        """)

# ==================== MAIN APP ====================
if not uploaded_file:
    # Welcome screen
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/144/000000/cloud.png", width=120)
    
    with col2:
        st.markdown(f"""
        ## 🚀 Smart Weather Classifier
        
        **Features:**
        - ✅ **Confidence threshold:** {confidence_threshold:.0%} minimum
        - ✅ **Rejects non-weather images**
        - ✅ **Compares {len(available_models)} strategies**
        - ✅ **Shows prediction time**
        
        **How it works:**
        1. Upload a weather image
        2. Model predicts with confidence score
        3. If confidence < {confidence_threshold:.0%} → **REJECTED**
        4. If confidence ≥ {confidence_threshold:.0%} → **ACCEPTED**
        
        **👈 Upload an image to start!**
        """)
    
    # Show confidence explanation
    with st.expander("🎯 Understanding Confidence Threshold", expanded=True):
        st.markdown(f"""
        | Confidence | Result | Meaning |
        |------------|--------|---------|
        | **< {confidence_threshold:.0%}** | ❌ **REJECTED** | "I'm not sure this is weather" |
        | **{confidence_threshold:.0%}-80%** | ⚠️ **LOW CONFIDENCE** | "Looks like weather, but not certain" |
        | **80%-95%** | ✅ **MEDIUM CONFIDENCE** | "This is probably weather" |
        | **> 95%** | 🏆 **HIGH CONFIDENCE** | "Definitely weather!" |
        
        **Current threshold:** {confidence_threshold:.0%}
        """)

else:
    # Load and validate image
    try:
        image = Image.open(uploaded_file)
        class_names = load_class_names()
        
        # Image validation
        validation_issues = validate_image(image)
        
        if validation_issues:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("⚠️ **Image Quality Issues:**")
            for issue in validation_issues:
                st.write(f"- {issue}")
            st.write("Prediction may be less accurate.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Uploaded Image")
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")
        
        with col2:
            st.subheader("🔍 Analysis")
            
            # Preprocess image
            img_processed = image.resize((224, 224))
            img_array = np.array(img_processed) / 255.0
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = np.expand_dims(img_array, axis=0)
            
            if not selected_names:
                st.warning("Please select at least one strategy from the sidebar!")
            else:
                # Load models
                models = load_models()
                
                # Check if we need feature extractor for ML models
                need_feature_extractor = any('SVM' in n or 'Random Forest' in n or 'MLP' in n for n in selected_names)
                feature_extractor = create_feature_extractor() if need_feature_extractor else None
                
                # Make predictions
                results = []
                rejected_models = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, name in enumerate(selected_names):
                    file = model_options[name]
                    
                    # Find the model
                    model_info = None
                    for key, val in models.items():
                        if key == name or val['file'] == file:
                            model_info = val
                            break
                    
                    if model_info:
                        status_text.text(f"Analyzing with {name}...")
                        
                        predictions, pred_time, max_conf = predict_with_rejection(
                            model_info, 
                            img_array, 
                            feature_extractor
                        )
                        
                        if predictions is None:
                            # Prediction rejected (low confidence)
                            rejected_models.append({
                                'strategy': name,
                                'time': pred_time,
                                'confidence': max_conf,
                                'reason': f"Confidence too low ({max_conf:.1%} < {confidence_threshold:.0%})"
                            })
                        else:
                            # Prediction accepted
                            pred_class = class_names[np.argmax(predictions)]
                            
                            results.append({
                                'strategy': name,
                                'class': pred_class,
                                'confidence': max_conf,
                                'time': pred_time,
                                'all_predictions': predictions,
                                'accuracy': load_accuracies().get(name, 0)
                            })
                        
                        progress_bar.progress((idx + 1) / len(selected_names))
                
                status_text.text("✅ Analysis complete!")
                
                # Display results
                if not results and not rejected_models:
                    st.error("❌ No models could process the image.")
                
                elif not results and rejected_models:
                    # ALL models rejected the image
                    st.markdown('<div class="reject-box">', unsafe_allow_html=True)
                    st.error("🚫 **IMAGE REJECTED**")
                    st.write(f"This doesn't appear to be a weather image.")
                    st.write(f"**Reason:** All models had confidence below {confidence_threshold:.0%}")
                    st.write("**Try:** Uploading a clearer weather image")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show rejection details
                    with st.expander("📋 Rejection Details"):
                        for reject in rejected_models:
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.write(f"**{reject['strategy']}**")
                            with col_b:
                                st.write(f"Confidence: {reject['confidence']:.1%}")
                            st.caption(f"Reason: {reject['reason']}")
                            st.progress(float(reject['confidence']))
                
                else:
                    # Some predictions were accepted
                    # Find best prediction
                    best = max(results, key=lambda x: x['confidence'])
                    
                    st.success(f"### 🏆 **{best['class'].upper()}**")
                    
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Confidence", f"{best['confidence']:.2%}")
                    with col_metric2:
                        st.metric("Strategy", best['strategy'])
                    with col_metric3:
                        st.metric("Time", f"{best['time']:.3f}s")
                    
                    # Show all accepted predictions
                    st.subheader("📊 Model Results")
                    
                    for result in results:
                        is_best = (result == best)
                        
                        with st.container():
                            col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
                            
                            with col_a:
                                if is_best:
                                    st.write(f"🏆 **{result['strategy']}**")
                                else:
                                    st.write(f"**{result['strategy']}**")
                            
                            with col_b:
                                st.write(f"`{result['class'].upper()}`")
                            
                            with col_c:
                                st.write(f"**{result['confidence']:.2%}**")
                            
                            with col_d:
                                st.write(f"`{result['time']:.3f}s`")
                            
                            # Confidence bar with color coding
                            confidence_color = (
                                "green" if result['confidence'] > 0.8 else
                                "orange" if result['confidence'] > confidence_threshold else
                                "red"
                            )
                            
                            st.progress(float(result['confidence']))
                    
                    # Show rejected models if any
                    if rejected_models:
                        st.subheader("❌ Rejected Predictions")
                        st.write(f"These models had confidence below {confidence_threshold:.0%}:")
                        
                        for reject in rejected_models:
                            col_x, col_y = st.columns([2, 1])
                            with col_x:
                                st.write(f"**{reject['strategy']}**")
                            with col_y:
                                st.write(f"Confidence: {reject['confidence']:.1%}")
                            st.progress(float(reject['confidence']))
                    
                    # Speed comparison chart
                    if len(results) > 1:
                        st.subheader("⏱️ Speed Comparison")
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[r['strategy'] for r in results],
                                y=[r['time'] for r in results],
                                text=[f"{r['time']:.3f}s" for r in results],
                                textposition='auto',
                                marker_color=['green' if r == best else 'blue' for r in results]
                            )
                        ])
                        
                        fig.update_layout(
                            title="Prediction Time by Strategy",
                            xaxis_title="Strategy",
                            yaxis_title="Time (seconds)",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence distribution
                        st.subheader("🎯 Confidence Distribution")
                        
                        fig2 = go.Figure(data=[
                            go.Scatter(
                                x=[r['strategy'] for r in results],
                                y=[r['confidence'] for r in results],
                                mode='markers+text',
                                marker=dict(
                                    size=[40 if r == best else 20 for r in results],
                                    color=['red' if r['confidence'] < confidence_threshold else 
                                          'orange' if r['confidence'] < 0.8 else 
                                          'green' for r in results]
                                ),
                                text=[f"{r['confidence']:.1%}" for r in results],
                                textposition="top center"
                            )
                        ])
                        
                        # Add threshold line
                        fig2.add_hline(
                            y=confidence_threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Threshold: {confidence_threshold:.0%}",
                            annotation_position="bottom right"
                        )
                        
                        fig2.update_layout(
                            title="Confidence Scores",
                            xaxis_title="Strategy",
                            yaxis_title="Confidence",
                            yaxis_range=[0, 1]
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try a different image file.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""
**Weather Classifier PRO** | Confidence Threshold: {confidence_threshold:.0%} | Sparse Fine-Tuning: 93.15% Accuracy
""")