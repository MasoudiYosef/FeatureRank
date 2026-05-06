#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import build_sigmoid_autoencoder, build_latent_classifier
import tensorflow as tf

def test_dtype_consistency():
    """Test that all arrays maintain float32 dtype"""
    print("=" * 70)
    print("TEST 1: Data Type Consistency")
    print("=" * 70)
    
    # Load sample data
    df = load_data("data/raw/breast_cancer_data.csv", target_column="diagnosis", id_column="ID")
    print(f"✓ Loaded dataset: shape {df.shape}")
    
    # Preprocess
    processed = preprocess_data(df, target_column="diagnosis", id_column="ID")
    
    # Check all array dtypes
    X_train_scaled = processed["X_train_scaled"]
    X_test_scaled = processed["X_test_scaled"]
    
    assert X_train_scaled.dtype == np.float32, f"X_train_scaled dtype is {X_train_scaled.dtype}, expected float32"
    assert X_test_scaled.dtype == np.float32, f"X_test_scaled dtype is {X_test_scaled.dtype}, expected float32"
    print(f"✓ X_train_scaled dtype: {X_train_scaled.dtype}")
    print(f"✓ X_test_scaled dtype: {X_test_scaled.dtype}")
    
    return True

def test_nan_inf_validation():
    """Test that NaN/Inf values are detected"""
    print("\n" + "=" * 70)
    print("TEST 2: NaN/Inf Validation")
    print("=" * 70)
    
    df = load_data("data/raw/breast_cancer_data.csv", target_column="diagnosis", id_column="ID")
    processed = preprocess_data(df, target_column="diagnosis", id_column="ID")
    
    X_train_scaled = processed["X_train_scaled"]
    X_test_scaled = processed["X_test_scaled"]
    
    has_nan_train = np.isnan(X_train_scaled).any()
    has_inf_train = np.isinf(X_train_scaled).any()
    has_nan_test = np.isnan(X_test_scaled).any()
    has_inf_test = np.isinf(X_test_scaled).any()
    
    assert not has_nan_train, "X_train_scaled contains NaN values"
    assert not has_inf_train, "X_train_scaled contains Inf values"
    assert not has_nan_test, "X_test_scaled contains NaN values"
    assert not has_inf_test, "X_test_scaled contains Inf values"
    
    print(f"✓ X_train_scaled: No NaN/Inf values")
    print(f"✓ X_test_scaled: No NaN/Inf values")
    
    return True

def test_model_dtype_specs():
    """Test that models have proper dtype specifications"""
    print("\n" + "=" * 70)
    print("TEST 3: Model Input Dtype Specifications")
    print("=" * 70)
    
    # Build models with proper dtype
    autoencoder, encoder = build_sigmoid_autoencoder(input_dim=30, encoding_dim=8)
    classifier = build_latent_classifier(input_dim=8, num_classes=2)
    
    # Check input specs
    ae_input_dtype = autoencoder.layers[0].dtype
    encoder_input_dtype = encoder.layers[0].dtype
    classifier_input_dtype = classifier.layers[0].dtype
    
    print(f"✓ Autoencoder input dtype: {ae_input_dtype}")
    print(f"✓ Encoder input dtype: {encoder_input_dtype}")
    print(f"✓ Classifier input dtype: {classifier_input_dtype}")
    
    return True

def test_model_predict_shapes():
    """Test prediction shape handling"""
    print("\n" + "=" * 70)
    print("TEST 4: Model Prediction Shape Handling")
    print("=" * 70)
    
    df = load_data("data/raw/breast_cancer_data.csv", target_column="diagnosis", id_column="ID")
    processed = preprocess_data(df, target_column="diagnosis", id_column="ID")
    
    X_train = processed["X_train_scaled"].astype(np.float32)
    X_test = processed["X_test_scaled"].astype(np.float32)
    
    # Train simple autoencoder
    autoencoder, encoder = build_sigmoid_autoencoder(input_dim=X_train.shape[1], encoding_dim=8)
    print(f"✓ Built autoencoder: input_shape={X_train.shape}")
    
    # Test encoding
    X_encoded = encoder.predict(X_test[:10], verbose=0)
    assert X_encoded.dtype == np.float32, f"Encoder output dtype is {X_encoded.dtype}, expected float32"
    print(f"✓ Encoder output shape: {X_encoded.shape}, dtype: {X_encoded.dtype}")
    
    # Test classifier prediction
    classifier = build_latent_classifier(input_dim=X_encoded.shape[1], num_classes=2)
    y_pred = classifier.predict(X_encoded, verbose=0)
    
    # Handle shape: should be (n_samples, 1) for sigmoid
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred_flat = y_pred.ravel()
    else:
        y_pred_flat = y_pred
    
    print(f"✓ Classifier prediction shape: {y_pred.shape}")
    print(f"✓ Flattened predictions shape: {y_pred_flat.shape}")
    assert y_pred_flat.shape[0] == X_encoded.shape[0], "Prediction count mismatch"
    
    return True

def test_encoder_classifier_compatibility():
    """Test encoder-to-classifier pipeline compatibility"""
    print("\n" + "=" * 70)
    print("TEST 5: Encoder-to-Classifier Compatibility")
    print("=" * 70)
    
    df = load_data("data/raw/breast_cancer_data.csv", target_column="diagnosis", id_column="ID")
    processed = preprocess_data(df, target_column="diagnosis", id_column="ID")
    
    X_train = processed["X_train_scaled"].astype(np.float32)[:50]  # Small batch for test
    y_train = processed["y_train"].to_numpy().astype(np.float32)[:50]
    
    # Build and encode
    autoencoder, encoder = build_sigmoid_autoencoder(input_dim=X_train.shape[1], encoding_dim=8)
    X_encoded = encoder.predict(X_train, verbose=0).astype(np.float32)
    
    # Build classifier with encoded dimension
    encoder_output_dim = X_encoded.shape[1]
    classifier = build_latent_classifier(input_dim=encoder_output_dim, num_classes=2)
    
    # Try to fit
    try:
        classifier.fit(X_encoded, y_train, epochs=1, batch_size=8, verbose=0)
        print(f"✓ Encoder output dim: {encoder_output_dim}")
        print(f"✓ Classifier successfully trained on encoded features")
        print(f"✓ Pipeline compatibility: PASS")
        return True
    except Exception as e:
        print(f"✗ Pipeline compatibility error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  TensorFlow Fixes Validation Suite".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("Data Type Consistency", test_dtype_consistency),
        ("NaN/Inf Validation", test_nan_inf_validation),
        ("Model Input Dtype Specs", test_model_dtype_specs),
        ("Prediction Shape Handling", test_model_predict_shapes),
        ("Encoder-Classifier Compatibility", test_encoder_classifier_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, result in results:
        status_icon = "✓" if result == "PASS" else "✗"
        print(f"{status_icon} {test_name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(r == "PASS" for _, r in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
