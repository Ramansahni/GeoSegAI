# GeoSegAI

AI-Based Eco-Aware Corridor Planning using Satellite Imagery.

## Features

- Vegetation segmentation using U-Net
- NDVI computation
- Environmental cost surface generation
- Interactive A* eco-routing
- Infrastructure-aware routing
- RGB overlay visualization

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run From

Run all scripts from inside the `src` directory.

---

## Baseline Pipeline

```bash
python infer.py
python cost_surface.py
python astar.py
python visualize.py
```

---

## Enhanced Pipeline

```bash
python road_prior.py
python enhanced_cost.py
python enhanced_astar.py
python enhanced_visualize.py
```