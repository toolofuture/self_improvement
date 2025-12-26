# Evollard Real Data Verification Algorithm (Self-Improving)
# Advisor: Gemini (based on advice from Prof. Ahn Doyp-Yeol)

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import glob
import subprocess

# Qiskit 2.0 / Primitives
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
import qiskit_algorithms.utils as alg_utils

# Configuration
PROJECT_ROOT = "/Users/kangsikseo/Downloads/evollard_quantum_prototype"
AUTHENTIC_DIR = "/Users/kangsikseo/Downloads/mt"
LEARNED_DIR = os.path.join(PROJECT_ROOT, "learned_knowledge")
TARGET_IMAGE = "/Users/kangsikseo/Downloads/DP-12769-001.jpg" 
# TRUTH_LABEL: 0 for Fake, 1 for Authentic. Set this to teach the AI.
TRUTH_LABEL = 1 

IMG_SIZE = (64, 64)
N_COMPONENTS = 2
alg_utils.algorithm_globals.random_seed = 12345
np.random.seed(12345)

print("=== Evollard Self-Improving Verification System Initializing... ===")

class SelfImprover:
    def __init__(self, learned_dir):
        self.learned_dir = learned_dir
        self.auth_dir = os.path.join(learned_dir, "authentic")
        self.fake_dir = os.path.join(learned_dir, "fake")
        os.makedirs(self.auth_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        print(f"[SelfImprover] Knowledge Base active at: {learned_dir}")
        self.sync_from_cloud()

    def sync_from_cloud(self):
        """Pulls the latest knowledge from GitHub."""
        print("[SelfImprover] ☁️ Syncing with Global Brain (GitHub)...")
        try:
            subprocess.run(["git", "pull", "origin", "main"], check=False, cwd=PROJECT_ROOT)
        except Exception as e:
            print(f"[SelfImprover] Sync Warning: {e}")

    def sync_to_cloud(self, message):
        """Pushes new knowledge to GitHub."""
        print("[SelfImprover] 🚀 Uploading new knowledge to Global Brain...")
        try:
            subprocess.run(["git", "add", "learned_knowledge/"], check=True, cwd=PROJECT_ROOT)
            subprocess.run(["git", "commit", "-m", message], check=True, cwd=PROJECT_ROOT)
            subprocess.run(["git", "push", "origin", "main"], check=True, cwd=PROJECT_ROOT)
            print("[SelfImprover] Upload Complete.")
        except Exception as e:
            print(f"[SelfImprover] Upload Failed: {e}")

    def load_knowledge(self):
        """Loads learned data from persistent storage."""
        print("[SelfImprover] Recalling learned patterns...")
        X_learned, y_learned = [], []
        
        # Load Authentic
        auth_files = glob.glob(os.path.join(self.auth_dir, "*"))
        for path in auth_files:
            try:
                data = process_image(path)
                X_learned.append(data)
                y_learned.append(1) # Authentic
            except: pass
            
        # Load Fake
        fake_files = glob.glob(os.path.join(self.fake_dir, "*"))
        for path in fake_files:
            try:
                data = process_image(path)
                X_learned.append(data)
                y_learned.append(0) # Fake
            except: pass
            
        print(f"[SelfImprover] Recalled {len(auth_files)} authentic and {len(fake_files)} fake patterns.")
        return X_learned, y_learned

    def teach(self, image_path, label):
        """Saves a verified image to the knowledge base."""
        target_dir = self.auth_dir if label == 1 else self.fake_dir
        filename = f"learned_{os.path.basename(image_path)}"
        dest_path = os.path.join(target_dir, filename)
        
        if not os.path.exists(dest_path):
            shutil.copy(image_path, dest_path)
            print(f"[SelfImprover] 🧠 LEARNING: Saved pattern from {os.path.basename(image_path)} to Knowledge Base.")
            print(f"               (Stored in: {target_dir})")
            
            # Auto-Push to GitHub
            self.sync_to_cloud(f"Self-Improvement: Learned {os.path.basename(image_path)} as Label {label}")
            
        else:
            print(f"[SelfImprover] I already know this pattern ({filename}).")

def process_image(path):
    with Image.open(path) as img:
        img = img.resize(IMG_SIZE).convert('L')
        return np.asarray(img).flatten()

def load_images_from_folder(folder, label, max_images=20):
    images = []
    labels = []
    count = 0
    print(f"[System] Loading up to {max_images} base images from {folder}...")
    
    valid_exts = ('.jpg', '.jpeg', '.png')
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])
    
    for filename in filenames:
        if count >= max_images:
            break
        img_path = os.path.join(folder, filename)
        try:
            data = process_image(img_path)
            images.append(data)
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            
    return np.array(images), np.array(labels)

def create_synthetic_fakes(authentic_data):
    print("[System] Generating Synthetic Fakes (Chaos Support)...")
    noise = np.random.normal(0, 50, authentic_data.shape)
    fakes = authentic_data + noise
    fakes = np.clip(fakes, 0, 255)
    return fakes, np.zeros(len(fakes))

# --- Main Execution Flow ---

# 1. Initialize Improver
improver = SelfImprover(LEARNED_DIR)

# 2. Load Base Data + Learned Data
X_real_base, y_real_base = load_images_from_folder(AUTHENTIC_DIR, label=1, max_images=30)
X_learned_list, y_learned_list = improver.load_knowledge()

# Combine lists to arrays
if X_learned_list:
    X_learned = np.array(X_learned_list)
    y_learned = np.array(y_learned_list)
    X_real = np.concatenate([X_real_base, X_learned[y_learned==1]]) if np.any(y_learned==1) else X_real_base
    # Note: learned fakes are handled separately below
else:
    X_real = X_real_base

# 3. Create/Load Fakes
X_fake_chaos, y_fake_chaos = create_synthetic_fakes(X_real_base) # Create chaos from base only
if X_learned_list and np.any(np.array(y_learned_list) == 0):
    X_fake_learned = np.array(X_learned_list)[np.array(y_learned_list) == 0]
    y_fake_learned = np.zeros(len(X_fake_learned))
    X_fake = np.concatenate([X_fake_chaos, X_fake_learned])
    y_fake = np.concatenate([y_fake_chaos, y_fake_learned])
else:
    X_fake, y_fake = X_fake_chaos, y_fake_chaos

# 4. Final Training Set
X_train = np.concatenate([X_fake, X_real])
y_train = np.concatenate([y_fake, np.ones(len(X_real))])

# 5. Preprocessing
print(f"[System] Training on {len(X_train)} samples (Authentic: {len(X_real)}, Fake: {len(X_fake)})...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=N_COMPONENTS)
X_train_pca = pca.fit_transform(X_train_scaled)

# 6. Quantum Kernel
print("[System] Computing Entanglement Kernel...")
feature_map = ZZFeatureMap(feature_dimension=N_COMPONENTS, reps=2, entanglement='linear')
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# 7. Training
qsvc = SVC(kernel=quantum_kernel.evaluate)
qsvc.fit(X_train_pca, y_train)

# 8. Verify & Auto-Improve
print(f"\n=== Verifying Target: {os.path.basename(TARGET_IMAGE)} ===")
try:
    with Image.open(TARGET_IMAGE) as img:
        img = img.resize(IMG_SIZE).convert('L')
        target_data = np.asarray(img).flatten().reshape(1, -1)
        
        target_scaled = scaler.transform(target_data)
        target_pca = pca.transform(target_scaled)
        
        prediction = qsvc.predict(target_pca)[0]
        
        result_text = "진품 (Authentic)" if prediction == 1 else "위작 (Fake / Chaos)"
        color = "\033[92m" if prediction == 1 else "\033[91m"
        reset = "\033[0m"
        
        print(f">>> Final Verdict: {color}[{result_text}]{reset}")
        print(f"    (Quantum Coordinates: {target_pca[0]})")
        
        # --- Self Improvement Trigger ---
        if TRUTH_LABEL is not None:
            if prediction != TRUTH_LABEL:
                print(f"\n[!] Mismatch detected! AI thought {prediction}, User says {TRUTH_LABEL}.")
                print("[!] Initiating Self-Correction...")
                improver.teach(TARGET_IMAGE, TRUTH_LABEL)
                print("[!] Please re-run the script to verify the correction.")
            else:
                print(f"\n[OK] AI Prediction matches Reality ({TRUTH_LABEL}).")
                # Reinforcement Learning: Teach it anyway to strengthen confidence?
                # For now, only teach on mismatch or explicit request.
                pass
        # --------------------------------

except Exception as e:
    print(f"Error processing target: {e}")
