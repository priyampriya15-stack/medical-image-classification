import urllib.request
import urllib.parse
import json
import uuid
import sys

BASE_URL = "http://127.0.0.1:8001/api"

def make_multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode('utf-8'))
    for name, (filename, content, content_type) in files.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode('utf-8'))
        body.extend(content)
        body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
    return boundary, bytes(body)

def test_endpoint(name, url, method="GET", data=None, headers=None, expected_status=200):
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=25) as res:
            res_body = res.read().decode('utf-8')
            print(f"[OK {res.status}] {name} ({len(res_body)} bytes)")
            try:
                parsed = json.loads(res_body)
                return True, parsed
            except Exception:
                return True, res_body
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        if e.code == expected_status:
            print(f"[OK {e.code}] {name} -> EXPECTED ERROR ({e.code})")
            return True, err_body
        else:
            print(f"[FAIL {e.code}] {name} -> {err_body[:200]}")
            return False, err_body
    except Exception as e:
        print(f"[EXC] {name} -> {e}")
        return False, str(e)

print("=== STARTING FULL PRODUCTION API SUITE CHECK ===")
results = {}

# 1. Health
ok, data = test_endpoint("1. Health Endpoint", f"{BASE_URL}/health")
results["health"] = ok

# 2. Project Metadata
ok, data = test_endpoint("2. Project Metadata", f"{BASE_URL}/project")
results["project_metadata"] = ok

# 3. Dataset Info
ok, data = test_endpoint("3. Dataset Info", f"{BASE_URL}/dataset")
results["dataset_info"] = ok
if ok and isinstance(data, dict):
    print(f"   -> Dataset: {data.get('name')} | Total Samples: {data.get('total_samples')} | Classes: {data.get('classes')}")

# 4. Dataset Samples
ok, data = test_endpoint("4. Dataset Samples", f"{BASE_URL}/dataset/samples?split=train&page=1&page_size=5")
results["dataset_samples"] = ok
if ok and isinstance(data, dict):
    print(f"   -> Samples returned: {len(data.get('items', []))} (total {data.get('total')})")

# 5. Models List
ok, data = test_endpoint("5. Models List", f"{BASE_URL}/models")
results["models_list"] = ok
if ok and isinstance(data, dict):
    for m in data.get('models', []):
        print(f"   -> Model: {m.get('name')} | Params: {m.get('parameters_count')} | Checkpoint: {m.get('checkpoint_available')}")

# 6. Model Detail
ok, data = test_endpoint("6. Model Detail (ResNet50)", f"{BASE_URL}/models/resnet50")
results["model_detail"] = ok

# 7. Model Detail (EfficientNetB0)
ok, data = test_endpoint("7. Model Detail (EfficientNetB0)", f"{BASE_URL}/models/efficientnet_b0")
results["model_detail_effnet"] = ok

# 8. Evaluation Comparison
ok, data = test_endpoint("8. Evaluation Comparison", f"{BASE_URL}/evaluation/comparison")
results["evaluation_comparison"] = ok
if ok and isinstance(data, dict):
    print(f"   -> Winner: {data.get('winner_model')} | Winner Metric: {data.get('winner_metric')}")

# 9. Evaluation Results
ok, data = test_endpoint("9. Evaluation Results (ResNet50)", f"{BASE_URL}/evaluation/results/resnet50")
results["evaluation_resnet50"] = ok
if ok and isinstance(data, dict):
    print(f"   -> ResNet50 Accuracy: {data.get('metrics', {}).get('accuracy'):.4f} | ROC-AUC: {data.get('metrics', {}).get('roc_auc'):.4f}")

# 10. Evaluation Results (EfficientNetB0)
ok, data = test_endpoint("10. Evaluation Results (EfficientNetB0)", f"{BASE_URL}/evaluation/results/efficientnet_b0")
results["evaluation_effnet"] = ok
if ok and isinstance(data, dict):
    print(f"   -> EfficientNetB0 Accuracy: {data.get('metrics', {}).get('accuracy'):.4f} | ROC-AUC: {data.get('metrics', {}).get('roc_auc'):.4f}")

# 11. Training Status
ok, data = test_endpoint("11. Training Status", f"{BASE_URL}/train/status")
results["training_status"] = ok

# 12. Analytics Summary
ok, data = test_endpoint("12. Analytics Summary", f"{BASE_URL}/analytics/summary")
results["analytics_summary"] = ok

# 13. Analytics ROC
ok, data = test_endpoint("13. Analytics ROC", f"{BASE_URL}/analytics/roc")
results["analytics_roc"] = ok

# 14. Analytics Confusion Matrix
ok, data = test_endpoint("14. Analytics Confusion Matrix", f"{BASE_URL}/analytics/confusion-matrix")
results["analytics_cm"] = ok

# 15. Analytics Precision-Recall
ok, data = test_endpoint("15. Analytics PR", f"{BASE_URL}/analytics/precision-recall")
results["analytics_pr"] = ok

# 16. Analytics Training History
ok, data = test_endpoint("16. Analytics Training History", f"{BASE_URL}/analytics/training-history")
results["analytics_history"] = ok

# 17. Activity Recent
ok, data = test_endpoint("17. Recent Activity", f"{BASE_URL}/activity")
results["activity_recent"] = ok

# 18. Real Inference: ResNet50 on Pneumonia radiograph
with open("backend/test_pneumonia.png", "rb") as f:
    pneumonia_bytes = f.read()

boundary, body = make_multipart(
    {"model_name": "resnet50"},
    {"image": ("test_pneumonia.png", pneumonia_bytes, "image/png")}
)
ok, data = test_endpoint("18. Predict ResNet50 (Pneumonia image)", f"{BASE_URL}/predict", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
results["predict_resnet50"] = ok
if ok and isinstance(data, dict):
    print(f"   -> Predicted: {data.get('class_name')} (class {data.get('predicted_class')}) | Conf: {data.get('confidence'):.4f} | Model: {data.get('model_name')}")

# 19. Real Inference: EfficientNetB0 on Normal radiograph
with open("backend/test_normal.png", "rb") as f:
    normal_bytes = f.read()

boundary, body = make_multipart(
    {"model_name": "efficientnet_b0"},
    {"image": ("test_normal.png", normal_bytes, "image/png")}
)
ok, data = test_endpoint("19. Predict EfficientNetB0 (Normal image)", f"{BASE_URL}/predict", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
results["predict_effnet"] = ok
if ok and isinstance(data, dict):
    print(f"   -> Predicted: {data.get('class_name')} (class {data.get('predicted_class')}) | Conf: {data.get('confidence'):.4f} | Model: {data.get('model_name')}")

# 20. Real Grad-CAM: ResNet50 (layer4)
boundary, body = make_multipart(
    {"model_name": "resnet50", "opacity": "0.65"},
    {"image": ("test_pneumonia.png", pneumonia_bytes, "image/png")}
)
ok, data = test_endpoint("20. Grad-CAM ResNet50 (layer4)", f"{BASE_URL}/gradcam", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
results["gradcam_resnet50"] = ok
if ok and isinstance(data, dict):
    has_overlay = bool(data.get("overlay_image"))
    has_heatmap = bool(data.get("heatmap_image"))
    print(f"   -> Overlay URI present: {has_overlay} (len={len(data.get('overlay_image', ''))}) | Heatmap URI: {has_heatmap} | Target: {data.get('target_layer')}")

# 21. Real Grad-CAM: EfficientNetB0 (features[-1])
boundary, body = make_multipart(
    {"model_name": "efficientnet_b0", "opacity": "0.65"},
    {"image": ("test_pneumonia.png", pneumonia_bytes, "image/png")}
)
ok, data = test_endpoint("21. Grad-CAM EfficientNetB0 (features[-1])", f"{BASE_URL}/gradcam", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
results["gradcam_effnet"] = ok
if ok and isinstance(data, dict):
    has_overlay = bool(data.get("overlay_image"))
    has_heatmap = bool(data.get("heatmap_image"))
    print(f"   -> Overlay URI present: {has_overlay} (len={len(data.get('overlay_image', ''))}) | Heatmap URI: {has_heatmap} | Target: {data.get('target_layer')}")

# 22. Error Handling: Unsupported model name
boundary, body = make_multipart(
    {"model_name": "vgg16_unknown"},
    {"image": ("test_normal.png", normal_bytes, "image/png")}
)
ok, data = test_endpoint("22. Error Handling: Invalid Model", f"{BASE_URL}/predict", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, expected_status=400)
results["error_invalid_model"] = ok

# 23. Error Handling: Corrupt / non-image file
boundary, body = make_multipart(
    {"model_name": "resnet50"},
    {"image": ("corrupted.txt", b"plain text data is not a valid radiograph image", "text/plain")}
)
ok, data = test_endpoint("23. Error Handling: Corrupted Image", f"{BASE_URL}/predict", method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, expected_status=400)
results["error_corrupt_image"] = ok

print("\n=== FINAL SUITE RESULTS SUMMARY ===")
all_passed = all(results.values())
for k, v in results.items():
    print(f"  {k:28}: {'PASS' if v else 'FAIL'}")
print(f"\nOVERALL RESULT: {'ALL 23 TESTS PASSED (100%)' if all_passed else 'SOME TESTS FAILED'}")
sys.exit(0 if all_passed else 1)
