---
title: Hybrid Medical AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hybrid Medical AI with HVF-Net verification
---

# Hybrid Medical AI + HVF-Net

Flask API that routes medical images to specialist models, then verifies with **HVF-Net** (accept / reject / re-route).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health + model status |
| POST | `/hybrid-predict` | Upload `image` (form-data) |

## Supported scans

- Brain MRI
- Chest X-ray
- Retina OCT

## Example

```bash
curl -X POST "https://YOUR_USERNAME-hybrid-medical-ai.hf.space/hybrid-predict" \
  -F "image=@sample.jpg"
```

Response includes `diagnosis`, `confidence`, `trust_score`, `hvf_action`, and `hvf_net`.
