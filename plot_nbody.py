import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results_nbody.csv",
                 names=["N", "cpu_time", "gpu_time", "block_size"])

# Merge CPU and GPU rows
cpu = df[df["cpu_time"].notna()][["N", "cpu_time"]]
gpu = df[df["gpu_time"].notna()][["N", "gpu_time"]]

data = pd.merge(cpu, gpu, on="N")

# Speedup
data["speedup"] = data["cpu_time"] / data["gpu_time"]

# ---- Plot 1: Time vs N ----
plt.figure()
plt.loglog(data["N"], data["cpu_time"], marker="o", label="CPU")
plt.loglog(data["N"], data["gpu_time"], marker="s", label="GPU (optimized)")
plt.xlabel("Number of particles (N)")
plt.ylabel("Time per step (s)")
plt.title("N-body Simulation Time per Step")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("time_vs_n.png")

# ---- Plot 2: Speedup ----
plt.figure()
plt.semilogx(data["N"], data["speedup"], marker="o")
plt.xlabel("Number of particles (N)")
plt.ylabel("Speedup (CPU / GPU)")
plt.title("GPU Speedup vs CPU")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_vs_n.png")

plt.show()
