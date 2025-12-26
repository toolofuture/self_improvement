# Evollard Quantum Verification Algorithm (Prototype)
# Advisor: Gemini (based on advice from Prof. Ahn Doyp-Yeol)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit import QuantumCircuit
# from qiskit_aer import AerSimulator # REMOVED: Incompatible with Python 3.14 for now

# Qiskit 2.0 / Primitives
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
import qiskit_algorithms.utils as alg_utils

# Ensure reproducibility
alg_utils.algorithm_globals.random_seed = 12345
np.random.seed(12345)

print("=== Evollard Quantum Verification System Initializing... ===")

def load_art_data():
    print("[System] Loading artwork feature data...")
    # Authentic (1): Centered around (0.5, 0.5)
    # Fake (0): Centered around (-0.5, -0.5)
    X_authentic = np.random.normal(0.5, 0.3, (10, 2))
    X_fake = np.random.normal(-0.5, 0.3, (10, 2))
    
    X = np.concatenate([X_fake, X_authentic])
    y = np.array([0]*10 + [1]*10) # 0: Fake, 1: Authentic
    return X, y

X_train, y_train = load_art_data()

print("[System] Preprocessing data (StandardScaler + PCA)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

print("[System] constructing Quantum Feature Map (ZZFeatureMap)...")
num_qubits = 2
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
print(f"=== Quantum Circuit Depth: {feature_map.depth()} ===")

# 4. 양자 커널 생성 (Using StatevectorSampler for local simulation without Aer)
print("[System] Computing Quantum Kernel Matrix using StatevectorSampler...")
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# 5. 분류기 학습
print("[System] Training QSVC (Quantum Support Vector Classifier)...")
qsvc = SVC(kernel=quantum_kernel.evaluate)
qsvc.fit(X_train_pca, y_train)

print(f"\n[Evollard System] Training Complete.")
print("System Ready for Verification.")

# --- 테스트 ---
print("\n=== Verifying New Artwork ===")
# Case A: Likely Authentic
X_test_authentic = np.array([[0.6, 0.7]]) 
# Case B: Likely Fake
X_test_fake = np.array([[-0.6, -0.4]])

X_test_authentic_trans = pca.transform(scaler.transform(X_test_authentic))
X_test_fake_trans = pca.transform(scaler.transform(X_test_fake))

try:
    pred_auth = qsvc.predict(X_test_authentic_trans)
    pred_fake = qsvc.predict(X_test_fake_trans)

    def print_result(name, prediction):
        result = "진품 (Authentic)" if prediction[0] == 1 else "위작 (Fake)"
        color_code = "\033[92m" if prediction[0] == 1 else "\033[91m"
        reset_code = "\033[0m"
        print(f"> Artwork '{name}': {color_code}[{result}]{reset_code}")

    print_result("Test Sample A", pred_auth)
    print_result("Test Sample B", pred_fake)

except Exception as e:
    print(f"Prediction failed: {e}")
    import traceback
    traceback.print_exc()

