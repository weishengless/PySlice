"""
Benchmark wrapper - runs each test in a fresh Python process to avoid Dask caching.
"""
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

# Test parameters
supercell_sizes = [8, 10, 12, 15, 18, 20, 25, 30]
probe_counts = [1, 4, 9, 16, 25, 49, 64, 100]

system_results = {'sizes': [], 'n_atoms': [], 'pyslice': [], 'abtem': [], 'speedups': []}
probe_results = {'counts': [], 'pyslice': [], 'abtem': [], 'speedups': []}

print("="*50)
print("SYSTEM SIZE SCALING (single probe)")
print("Running each test in fresh Python process")
print("="*50)

for size in supercell_sizes:
    print(f"\n--- Testing {size}³ supercell ---")

    # Run in fresh Python process
    result = subprocess.run(
        ['python3', '09_benchmark_single.py', 'system', str(size)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        continue

    # Read results from file
    with open(f'result_system_{size}.txt', 'r') as f:
        line = f.read().strip()
        size_val, n_atoms, pyslice_time, abtem_time = line.split(',')
        n_atoms = int(n_atoms)
        pyslice_time = float(pyslice_time)
        abtem_time = float(abtem_time)

    print(f"  {n_atoms} atoms")
    print(f"  PySlice: {pyslice_time:.3f}s")
    print(f"  AbTem: {abtem_time:.3f}s")

    speedup = abtem_time / pyslice_time
    print(f"  Speedup: {speedup:.2f}×")

    system_results['sizes'].append(int(size_val))
    system_results['n_atoms'].append(n_atoms)
    system_results['pyslice'].append(pyslice_time)
    system_results['abtem'].append(abtem_time)
    system_results['speedups'].append(speedup)

print("\n" + "="*50)
print("MULTI-PROBE SCALING (10³ supercell)")
print("Running each test in fresh Python process")
print("="*50)

for n_probes in probe_counts:
    print(f"\n--- Testing {n_probes} probes ---")

    # Run in fresh Python process
    result = subprocess.run(
        ['python3', '09_benchmark_single.py', 'probe', str(n_probes)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        continue

    # Read results from file
    with open(f'result_probe_{n_probes}.txt', 'r') as f:
        line = f.read().strip()
        n_probes_val, pyslice_time, abtem_time = line.split(',')
        n_probes_val = int(n_probes_val)
        pyslice_time = float(pyslice_time)
        abtem_time = float(abtem_time)

    print(f"  PySlice: {pyslice_time:.3f}s ({pyslice_time/n_probes:.4f}s/probe)")
    print(f"  AbTem: {abtem_time:.3f}s ({abtem_time/n_probes:.4f}s/probe)")

    speedup = abtem_time / pyslice_time
    print(f"  Speedup: {speedup:.2f}×")

    probe_results['counts'].append(n_probes_val)
    probe_results['pyslice'].append(pyslice_time)
    probe_results['abtem'].append(abtem_time)
    probe_results['speedups'].append(speedup)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# System size scaling
if system_results['sizes']:
    ax1.plot(system_results['sizes'], system_results['pyslice'], 'o-', linewidth=2, markersize=8, label='PySlice')
    ax1.plot(system_results['sizes'], system_results['abtem'], 's-', linewidth=2, markersize=8, label='AbTem')
    ax1.set_xlabel('Supercell Size', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('System Size Scaling (1 probe)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Multi-probe scaling
if probe_results['counts']:
    ax2.plot(probe_results['counts'], probe_results['pyslice'], 'o-', linewidth=2, markersize=8, label='PySlice')
    ax2.plot(probe_results['counts'], probe_results['abtem'], 's-', linewidth=2, markersize=8, label='AbTem')
    ax2.set_xlabel('Number of Probes', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Multi-Probe Scaling (10³ supercell)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_no_cache.png', dpi=300)
print("\nPlot saved to benchmark_no_cache.png")
plt.show()

# Print summary
print("\n" + "="*50)
print("BENCHMARK RESULTS")
print("="*50)

if system_results['sizes']:
    print("\nSYSTEM SIZE SCALING (single probe):")
    print(f"{'Size':>5} {'Atoms':>6} {'PySlice(s)':>11} {'AbTem(s)':>10} {'Speedup':>8}")
    print("-"*45)
    for i in range(len(system_results['sizes'])):
        print(f"{system_results['sizes'][i]:>5} {system_results['n_atoms'][i]:>6} "
              f"{system_results['pyslice'][i]:>10.3f} {system_results['abtem'][i]:>9.3f} "
              f"{system_results['speedups'][i]:>7.2f}×")
    print(f"Average speedup: {np.mean(system_results['speedups']):.2f}×")

if probe_results['counts']:
    print("\nMULTI-PROBE SCALING (10³ supercell):")
    print(f"{'Probes':>6} {'PySlice(s)':>11} {'AbTem(s)':>10} {'Speedup':>8}")
    print("-"*40)
    for i in range(len(probe_results['counts'])):
        print(f"{probe_results['counts'][i]:>6} {probe_results['pyslice'][i]:>10.3f} "
              f"{probe_results['abtem'][i]:>9.3f} {probe_results['speedups'][i]:>7.2f}×")
    print(f"Average speedup: {np.mean(probe_results['speedups']):.2f}×")
