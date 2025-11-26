import numpy as np

NPZ_PATH = "turing_patterns_dataset_cupy_18000_20000.npz"

data = np.load(NPZ_PATH)
print("Keys:", data.files)

u = data["u"]
v = data["v"]
ids = data["ids"]
a = data["a"]
delta = data["delta"]

print("u shape:", u.shape)       # (N, 128, 128)
print("v shape:", v.shape)
print("ids shape:", ids.shape)
print("a shape:", a.shape)
print("delta shape:", delta.shape)

print("前 5 個 id:", ids[:5])
print("前 5 個 a:", a[:5])
print("前 5 個 delta:", delta[:5])
