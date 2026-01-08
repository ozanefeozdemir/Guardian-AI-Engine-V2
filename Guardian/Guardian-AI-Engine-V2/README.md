# 🛡️ Guardian AI Engine V2 - Advanced Cybersecurity AI System

**Guardian AI Engine V2** is a real-time, machine learning-based, adaptive Intrusion Detection System (IDS).

This system utilizes a **Random Forest** model trained on **CIC-IDS 2017/2018** datasets to detect cyber attacks (DDoS, Brute Force, Web Attacks, Botnets, etc.) with up to **99.8% accuracy**. Its key feature is **Online Learning**, allowing it to learn network characteristics and adapt continuously to new threats.

---

## 🏗️ Architecture

The project employs a **Microservices** architecture for performance and scalability:

1.  **🧠 Analysis Engine (The Worker):**
    *   Listens to live traffic or reads simulation files.
    *   Analyzes every packet using the AI model.
    *   Uses the first 5% of incoming data to retrain and adapt itself to the specific day's attack patterns (Adaptation).
    *   Pushes results to the Redis queue.
2.  **⚡ Redis (Message Queue):**
    *   Acts as an ultra-fast communication bridge between the Engine and the API, preventing data loss.
3.  **📡 API Server (FastAPI):**
    *   Consumes attack alerts from Redis.
    *   Broadcasting real-time data to the Live Dashboard via WebSockets.
4.  **🖥️ Live Dashboard (Frontend):**
    *   Visualizes attacks and benign traffic with real-time graphs.

---

## 🚀 Installation & Usage

### Method 1: Docker (Recommended)

Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed.

1.  **Open terminal and navigate to the project directory.**
2.  **Start the system:**
    ```bash
    docker-compose up --build
    ```
    *This command starts Redis, API, and Analysis Engine services in order.*


---

### Method 2: Manual (Local) Execution

For development purposes, you can run services individually.
*(Requirements: Python 3.10+, Redis Server)*

1.  **Setup Virtual Environment:**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **Start Redis:**
    *   Ensure a Redis server is running on port 6379.

4.  **Start API Server (Terminal 1):**
    ```powershell
    python backend/api.py
    ```

5.  **Start Engine (Terminal 2):**
    ```powershell
    # Runs in simulation mode
    python backend/analyze_engine.py --mode simulation
    ```

---

## 📂 File Structure

*   `backend/` - All Python source code.
    *   `analyze_engine.py`: Core engine. Reads traffic, queries AI model, writes to Redis.
    *   `train_model.py`: Tool for training and testing the model.
    *   `api.py`: WebSocket server.
    *   `feature_extractor.py`: Converts raw packet data (IP, Port, etc.) into numerical features for the model.
    *   `datasets/`: Training and testing CSV files.
    *   `saved_models/`: Trained AI model files (`.pkl`).
*   `live_monitor.html`: Real-time dashboard.
*   `docker-compose.yml`: Orchestration configuration.

---

## 🛠️ Training & Testing the Model

If you wish to retrain the model from scratch or test with a new dataset, use `train_model.py`.

**Train Base Model (with 2017 Data):**
```powershell
python backend/train_model.py --mode train_base
```

**Run Adaptation Test (with 2018 Data):**
```powershell
# --file is optional, uses default if omitted.
python backend/train_model.py --mode adapt --file "backend/datasets/raw/CIC-IDS 2018/03-01-2018.csv"
```

---

## ❓ FAQ

**Q: Why does the IP show as "Unknown" in logs?**
A: If you are using the "Machine Learning" version of the datasets, columns like sensitive IPs are removed for privacy. Using "PCAP" or "Traffic Lab" versions will reveal the IPs.

**Q: What is the Confidence score?**
A: It indicates how certain the AI is about its decision. 0.99 means the model is 99% sure it's an attack.

**Q: Why does the system sometimes skip adaptation?**
A: If the test file contains NO attacks (only Benign traffic), the system detects this and decides there is nothing new to learn, skipping adaptation to preserve the base model's integrity. This is a safety feature.
