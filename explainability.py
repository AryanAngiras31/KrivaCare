import torch
import numpy as np
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, image_tensor, original_image_np):
    """
    Generates a Grad-CAM heatmap for the ultrasound image.
    original_image_np must be normalized between 0 and 1.
    """
    # Target the final convolutional layer of ResNet18
    target_layers = [model.backbone.layer4[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate map for the predicted class
    targets = [ClassifierOutputTarget(torch.argmax(model(image_tensor)).item())]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    
    # Overlay heatmap on original image
    visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    return visualization

def generate_shap_values(tabular_model, background_data, patient_data):
    """
    Generates SHAP values for the tabular features.
    background_data: A representative sample of the training set to establish baselines.
    """
    tabular_model.eval()
    
    # Initialize the Deep Explainer
    explainer = shap.DeepExplainer(tabular_model, background_data)
    
    # Calculate SHAP values for the specific patient
    shap_values = explainer.shap_values(patient_data)
    
    return shap_values