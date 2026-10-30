# 🚦 AUTONOMOUS SMART TRAFFIC MANAGEMENT

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0%2B-6929C4.svg)](https://qiskit.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLO%20%7C%20CNN%20%7C%20LSTM-orange.svg)]()
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, autonomous urban traffic control system combining **Quantum Computing (Qiskit QAOA)**, **Deep Learning (YOLO + CNN + LSTM)**, and **real-time geospatial maps (Leaflet + Google Maps Distance Matrix API)** to dynamically optimize traffic signal phases and clearance times without static clocks.

---

## 📌 Key Architectural Pillars

### 1. ⚛️ Quantum Approximate Optimization Algorithm (QAOA)
- Multi-intersection coordination is an **NP-hard** combinatorial optimization problem ($O(2^N)$ states).
- Formulates a **Quadratic Unconstrained Binary Optimization (QUBO)** model mapped to an **Ising Hamiltonian**:
  $$H(z) = \sum_{i} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j$$
- **Linear terms ($h_i$)**: Encode directional queue differentials between North-South and East-West approaches, accumulated wait times, and emergency priority.
- **Coupling terms ($J_{ij}$)**: Couple **current**, **next**, and **previous** traffic lights along arterial corridors to establish continuous green waves and suppress downstream spillback.
- **5-Qubit QAOA Circuit ($p=1$)**: Built in Qiskit using Hadamard initialization, $R_Z$ problem unitaries, $CX$ entanglers, and $R_X$ mixers. Statevector simulation converges onto optimal phase bitstrings in $<15\text{ ms}$.
- **Autonomous Signal Durations**: Dynamically adjusted between **8 seconds** (safety threshold) and **70 seconds** (rush-hour surge) based on queue pressure.

### 2. 🧠 Deep Learning Suite (YOLO + 1D-CNN + 2-Layer LSTM)
- **Vehicle Detection & Classification (YOLOv8)**:
  - Identifies **cars, buses, trucks, bikes, and autos** across intersections.
  - Computes weighted Passenger Car Equivalents (PCE) where heavy vehicles create significantly higher congestion:
    - 🚛 **Truck**: $3.5\times$
    - 🚌 **Bus**: $3.0\times$
    - 🚗 **Car**: $1.0\times$
    - 🛺 **Auto**: $0.8\times$
    - 🏍️ **Bike**: $0.5\times$
- **Spatial Traffic Density Classifier (1D-CNN)**:
  - Ingests 20-dimensional intersection feature vectors (type distributions, queue lengths, approach velocities, time slots).
  - Categorizes traffic into 4 discrete congestion states: `Low`, `Medium`, `High`, and `Critical`.
- **Temporal Congestion Forecasting (2-Layer LSTM)**:
  - 64 hidden units with recurrent dropout (0.2).
  - Takes 12-step historical traffic sequences and outputs multi-step forecasts: **+15m, +30m, +45m, and +60m**.
- **Datasets**: Compatible with open benchmark traffic datasets (**PeMS-BAY** and **METR-LA**) alongside an automated 30-day synthetic traffic generation pipeline for Indian urban corridors.

### 3. 🕒 3-Hour Time Slots & Google Maps Integration
- Calibrates traffic density dynamically across eight 3-hour diurnal windows:
  - `00:00 - 03:00`: Late Night (15% capacity)
  - `03:00 - 06:00`: Early Morning (20% capacity)
  - `06:00 - 09:00`: Morning Rush (90% capacity)
  - `09:00 - 12:00`: Late Morning (65% capacity)
  - `12:00 - 15:00`: Afternoon (55% capacity)
  - `15:00 - 18:00`: Evening Rush (95% capacity)
  - `18:00 - 21:00`: Evening (70% capacity)
  - `21:00 - 00:00`: Night (30% capacity)
- **Google Maps API**: Optional runtime connection to the Google Maps Distance Matrix and Directions APIs with a calibrated fallback model when no API key is specified.

### 4. 🚑 Instant Emergency Green Corridors
- Immediate preemption for **Ambulances** and **Fire Trucks**.
- Computes Dijkstra's shortest path across the city intersection graph.
- Instantly forces **GREEN** along the emergency vehicle's clearance corridor while setting all conflicting lights to **RED** ($w_{\text{emerg}} = 500.0$ in the QAOA Hamiltonian).
- Automatically clears preemption upon corridor exit.

### 5. 🗺️ Real City Road Networks & Interactive UI/UX
- Replicates 5 Indian metropolitan networks with real GPS coordinates:
  - **Bhopal** (DB Mall, New Market, Habibganj Station, MP Nagar, ISBT)
  - **Gwalior** (Phool Bagh, Maharaj Bada, Jayendraganj, City Center, Bada Chowk)
  - **New Delhi** (ITO Junction, Connaught Place, AIIMS, Karol Bagh, Rajiv Chowk)
  - **Bangalore** (Silk Board, KR Puram, Hebbal Flyover, Koramangala, MG Road)
  - **Chandigarh** (Sector 17, ISBT-43, Tribune Chowk, Sector 22, PGI Roundabout)
- Visualizes 3-5 traffic lights simultaneously with clear **RED/GREEN** glow states, directional queue pills, and digital countdown timers.
- Displays live **Earlier (Fixed Timer) vs. Current (Quantum QAOA)** wait times showing ~35% - 55% reduction.

---

## 📁 Repository Structure

```
autonomous_smart_traffic/
├── backend/                        # Python Flask Server & Core Engines
│   ├── app.py                      # REST API & background simulation loop
│   ├── city_networks.py            # GPS coordinates & graph topology for 5 cities
│   ├── config.py                   # Central constants, weights, and hyperparameters
│   ├── emergency_handler.py        # Dijkstra shortest path & corridor preemption
│   ├── google_maps_service.py      # Google Maps Distance Matrix & fallback model
│   ├── quantum_engine.py           # Qiskit QAOA 5-qubit solver & QUBO formulation
│   └── traffic_simulator.py        # Microscopic traffic simulator & IDM dynamics
│
├── dl_model/                       # Deep Learning Suite (Train, Test, Data)
│   ├── models/
│   │   ├── congestion_lstm.py      # 2-layer LSTM time-series forecasting model
│   │   ├── vehicle_cnn.py          # 1D-CNN traffic density classifier
│   │   └── yolo_detector.py        # YOLOv8 fleet detector with heavy-vehicle weighting
│   ├── train/
│   │   ├── config.py               # DL training hyperparameters & split configurations
│   │   ├── train_cnn.py            # CNN training routine with early stopping
│   │   └── train_lstm.py           # LSTM training routine with MSE loss
│   ├── test/
│   │   ├── test_cnn.py             # CNN accuracy, precision, recall, confusion matrix
│   │   ├── test_lstm.py            # LSTM MAE, RMSE & classification evaluation
│   │   └── test_yolo.py            # Detection verification & weighted PCE tests
│   └── data/
│       ├── download_dataset.py     # PeMS-BAY & METR-LA downloader
│       ├── generate_synthetic.py   # 30-day realistic traffic dataset synthesizer
│       ├── preprocess.py           # Normalization & sliding-window sequence builder
│       ├── raw/                    # Raw traffic benchmark datasets
│       └── processed/              # Formatted training arrays (.npz)
│
├── frontend/                       # Interactive Command Center Dashboard
│   ├── templates/
│   │   └── index.html              # Multi-panel HTML5 interface
│   └── static/
│       ├── css/
│       │   └── style.css           # Dark command center styling & signal animations
│       └── js/
│           ├── app.js              # Application orchestrator & 5Hz polling loop
│           ├── dl_panel.js         # DL inference visualization & fleet mix doughnut
│           ├── emergency.js        # Emergency dispatch UI & corridor rendering
│           ├── map_renderer.js     # Leaflet multi-lane road & intersection renderer
│           ├── quantum_panel.js    # QAOA circuit inspector & QUBO coupling heatmap
│           ├── traffic_lights.js   # Dual-phase signal lights & countdown displays
│           └── vehicle_layer.js    # Real-time vehicle markers & congestion meter
│
├── .gitignore                      # Git exclusion rules
├── LICENSE                         # MIT License
├── README.md                       # Complete Project Documentation
├── requirements.txt                # Python dependencies
├── run.bat                         # One-click Windows execution launcher
└── run.py                          # Cross-platform entrypoint & auto-browser launcher
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Web browser (Chrome, Edge, Firefox, or Safari)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AUTONOMOUS-SMART-TRAFFIC-MANAGEMENT.git
cd AUTONOMOUS-SMART-TRAFFIC-MANAGEMENT

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the System
```bash
python run.py
```
*Or on Windows, double-click `run.bat`.*

The dashboard will open automatically in your browser at:
```
http://127.0.0.1:5000
```

---

## 🧪 Deep Learning Pipeline (Train & Test)

All DL components support standalone execution and include fallback execution modes:

```bash
# Generate synthetic traffic datasets for all 5 Indian cities
python -m dl_model.data.generate_synthetic

# Preprocess raw sequences for training
python -m dl_model.data.preprocess

# Train the CNN density classifier
python -m dl_model.train.train_cnn

# Train the 2-layer LSTM congestion forecasting model
python -m dl_model.train.train_lstm

# Run test evaluations
python -m dl_model.test.test_cnn
python -m dl_model.test.test_lstm
python -m dl_model.test.test_yolo
```

---

## 🌐 REST API Specifications

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves the main Command Center UI |
| `GET` | `/api/state` | Returns complete live state (intersections, lights, vehicles, quantum, DL) |
| `GET` | `/api/cities` | Returns available city networks and coordinates |
| `POST` | `/api/city` | Switches active city: `{"city_id": "bhopal"}` |
| `POST` | `/api/time_slot` | Switches 3-hour traffic slot: `{"slot_id": "06-09"}` |
| `POST` | `/api/mode` | Toggles control mode: `{"mode": "quantum" \| "classical"}` |
| `POST` | `/api/emergency` | Dispatches emergency vehicle: `{"type": "ambulance" \| "firetruck"}` |
| `POST` | `/api/emergency/clear`| Dismisses active emergency preemption |
| `POST` | `/api/burst` | Injects traffic congestion surge: `{"count": 20}` |
| `POST` | `/api/quantum_solve`| Forces manual QAOA quantum optimization |
| `POST` | `/api/google_key` | Updates Google Maps API key dynamically: `{"api_key": "AIza..."}` |
| `POST` | `/api/playback` | Controls simulation play/pause and speed factor |

---

## 🚀 How to Upload to GitHub

Follow these steps in your terminal inside the project directory:

```bash
# 1. Initialize Git repository
git init

# 2. Add all project files
git add .

# 3. Commit your code
git commit -m "feat: Autonomous Smart Traffic Management System with Quantum QAOA and Deep Learning"

# 4. Set main branch
git branch -M main

# 5. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/AUTONOMOUS-SMART-TRAFFIC-MANAGEMENT.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 📄 License
This project is open-source and distributed under the [MIT License](LICENSE).
