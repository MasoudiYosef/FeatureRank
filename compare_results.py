#!/usr/bin/env python3
"""
Multi-class dataset'lerin accuracy sonuçlarını karşılaştır (%20, %30, %40 vs)
"""

import json
from pathlib import Path

output_dir = Path("outputs/autoencoder")
results = {}

# Her dataset'i bul
for dataset_dir in sorted(output_dir.iterdir()):
    if not dataset_dir.is_dir():
        continue
    
    dataset_name = dataset_dir.name
    metrics_dir = dataset_dir / "metrics"
    
    if not metrics_dir.exists():
        continue
    
    org_file = metrics_dir / "ORG_test_metrics.json"
    if not org_file.exists():
        continue
    
    with open(org_file) as f:
        org_data = json.load(f)
    
    # Multi-class olup olmadığını kontrol et
    num_classes = org_data.get("num_classes", 1)
    if num_classes <= 2:
        continue  # Binary dataset'i atla
    
    org_acc = org_data.get("test_accuracy", "N/A")
    
    # Farklı feature yüzdeleri için sonuçları topla
    feature_results = {"ORG": org_acc}
    
    for metric_file in sorted(metrics_dir.glob("top_*_test_metrics.json")):
        with open(metric_file) as f:
            data = json.load(f)
        percent = int(data.get("feature_percent", 0))
        acc = data.get("test_accuracy", "N/A")
        feature_results[percent] = acc
    
    results[dataset_name] = {
        "num_classes": num_classes,
        "results": feature_results
    }

# Tablo olarak yazdır
print("\n" + "="*90)
print("MULTI-CLASS DATASET'LERİ: ACCURACY KARŞILAŞTIRMASI")
print("="*90 + "\n")

for dataset, data in sorted(results.items()):
    num_classes = data["num_classes"]
    accs = data["results"]
    
    print(f"\n{dataset} ({num_classes} sınıf):")
    print("-" * 60)
    
    # ORG
    if "ORG" in accs:
        org_acc = accs["ORG"]
        print(f"  ORG:     {org_acc:.6f}")
    
    # Yüzdeleri sırayla
    for percent in sorted([k for k in accs.keys() if isinstance(k, int)]):
        acc = accs[percent]
        if isinstance(acc, float):
            org_val = accs.get("ORG", 0)
            diff = acc - org_val
            diff_str = f"{diff:+.6f}" if isinstance(diff, float) else "?"
            symbol = "↑" if diff > 0.0001 else "↓" if diff < -0.0001 else "="
            print(f"  %{percent:3d}:  {acc:.6f}  ({symbol} {diff_str})")
        else:
            print(f"  %{percent:3d}:  {acc}")

print("\n" + "="*90)
