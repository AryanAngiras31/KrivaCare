import torch
import torch.nn.functional as F

def calculate_entropy(probs):
    """Calculates Shannon Entropy for a probability distribution."""
    # Add epsilon to prevent log(0)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def uncertainty_aware_fusion(logits_img, logits_tab):
    """Dynamically fuses image and tabular predictions based on confidence."""
    # Convert raw logits to probabilities
    p_img = F.softmax(logits_img, dim=1)
    p_tab = F.softmax(logits_tab, dim=1)

    # Calculate entropy (uncertainty) for each modality
    entropy_img = calculate_entropy(p_img)
    entropy_tab = calculate_entropy(p_tab)

    # Calculate dynamic weights using exponential negation of entropy
    # Lower entropy -> Higher weight
    weight_img = torch.exp(-entropy_img)
    weight_tab = torch.exp(-entropy_tab)

    # Normalize weights so they sum to 1
    total_weight = weight_img + weight_tab
    w_img_norm = (weight_img / total_weight).unsqueeze(1)
    w_tab_norm = (weight_tab / total_weight).unsqueeze(1)

    # Fuse probabilities
    fused_probs = (w_img_norm * p_img) + (w_tab_norm * p_tab)
    
    # Get final class prediction
    final_prediction = torch.argmax(fused_probs, dim=1)
    
    return fused_probs, final_prediction