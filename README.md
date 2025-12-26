# Evollard Quantum Verification System (Self-Improving)

This project implements a **Quantum Art Verification Algorithms** that learns from feedback using an **Auto-Syncing Active Learning** mechanism.

## 🚀 Features
- **Quantum Entanglement Verification**: Uses `ZZFeatureMap` (Entanglement) to distinguish authentic art from chaos/forgeries.
- **Self-Improvement**: Automatically saves user-verified patterns to a persistent knowledge base.
- **Hive Mind (Git Sync)**: Automatically pulls/pushes learned knowledge to GitHub (`toolofuture/self_improvement`) to share intelligence across devices.

## 🛠️ Setup (Run this once)

### 1. Clone the Repository
```bash
git clone https://github.com/toolofuture/self_improvement.git
cd self_improvement
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. [IMPORTANT] Prepare the Dataset ⚠️
This repository **does NOT contain the base authentic images** due to copyright/size.
You must manually place your authentic image folder (e.g., `mt`) in `Downloads` or update the script path.

- **Required Path**: `/Users/kangsikseo/Downloads/mt` (Default)
- *If your path is different, open `evollard_real_verification.py` and update line 21: `AUTHENTIC_DIR = "..."`*

### 4. Configure GitHub Token (Optional but Recommended)
For automatic syncing to work, ensure your Git is configured with the Personal Access Token (PAT).
```bash
git remote set-url origin https://<YOUR_PAT>@github.com/toolofuture/self_improvement.git
```

## 🕵️‍♂️ Usage

1. Open `evollard_real_verification.py`.
2. Set the `TARGET_IMAGE` variable to the file you want to verify.
3. (Optional) Set `TRUTH_LABEL` (1=Authentic, 0=Fake) if you want to **teach** the AI.
4. Run the script:
   ```bash
   python evollard_real_verification.py
   ```

## 📁 Structure
- `evollard_real_verification.py`: Main executable (Classification + Self-Improvement).
- `learned_knowledge/`: The "Brain" of the AI. Contains patterns learned from user feedback. **(Synced with GitHub)**.
- `requirements.txt`: Python package list.
