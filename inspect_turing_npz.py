import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "turing_patterns_dataset_merged.npz"

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

# 繪製圖靈斑紋
num_samples = min(6, len(u))  # 顯示前 6 個樣本

fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
if num_samples == 1:
    axes = axes.reshape(2, 1)

for i in range(num_samples):
    # 繪製 u 場
    axes[0, i].imshow(u[i], cmap='RdBu_r', interpolation='bilinear')
    axes[0, i].set_title(f'u - ID:{ids[i]}\na={a[i]:.4f}, δ={delta[i]:.4f}', fontsize=8)
    axes[0, i].axis('off')
    
    # 繪製 v 場
    axes[1, i].imshow(v[i], cmap='RdBu_r', interpolation='bilinear')
    axes[1, i].set_title(f'v - ID:{ids[i]}', fontsize=8)
    axes[1, i].axis('off')

plt.suptitle('圖靈斑紋 (Turing Patterns)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
