import sys,os
sys.path.insert(1,"../../")
from src.multislice.potentials import Potential
from src.multislice.multislice import Probe, Propagate, create_batched_probes
from src.multislice.trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt
import time
from ase.build import bulk
import gc,shutil

# Check for device availability
try:
    import torch

    # Disable Dask caching globally BEFORE any AbTem operations
    import dask
    dask.config.set(scheduler='synchronous')
    dask.config.set({'array.cache': None})  # Disable array caching

    import abtem

    # Detect available devices
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if has_cuda:
        device = 'cuda'
        abtem_device = 'gpu'
        force_cpu = False
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("Using: CUDA GPU (fair comparison)\n")
    elif has_mps:
        # Force both to use CPU for fair comparison
        device = 'cpu'
        abtem_device = 'cpu'
        force_cpu = True
        print("Device: Apple Silicon detected")
        print("AbTem doesn't support MPS, so forcing BOTH to CPU for fair comparison")
        print("Using: CPU only\n")
    else:
        device = 'cpu'
        abtem_device = 'cpu'
        force_cpu = False
        print("Using: CPU only\n")

except ImportError:
    print("PyTorch or AbTem not available")
    sys.exit(1)

# Test parameters
supercell_sizes = [8, 10, 12, 15, 18, 20, 25, 30]  # System size scaling
probe_counts = [1, 4, 9, 16, 25, 49, 64, 100]  # Multi-probe scaling

def create_trajectory(supercell_size):
    """Create Cu supercell trajectory."""
    atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
    atoms *= (supercell_size, supercell_size, supercell_size)

    box_x = atoms.cell[0,0]
    box_y = atoms.cell[1,1]
    box_z = atoms.cell[2,2]

    trajectory = Trajectory(
        positions=atoms.positions.reshape(1, -1, 3),
        velocities=np.zeros((1, len(atoms), 3)),
        box_matrix=np.diag([box_x, box_y, box_z]),
        atom_types=np.array([29] * len(atoms)),
        timestep=1.0
    )
    return trajectory, atoms

# Generate probe grid positions
def probe_positions(box_x, box_y, n_probes):
    if n_probes == 1:
        return [(box_x/2, box_y/2)]
    n_side = int(np.ceil(np.sqrt(n_probes)))
    positions = []
    xs = np.linspace(box_x*0.2, box_x*0.8, n_side)
    ys = np.linspace(box_y*0.2, box_y*0.8, n_side)
    for x in xs:
        for y in ys:
            positions.append((x, y))
            if len(positions) >= n_probes:
                break
        if len(positions) >= n_probes:
            break
    return positions[:n_probes]

# Storage for results
system_results = {'sizes': [], 'n_atoms': [], 'pyslice': [], 'abtem': [], 'speedups': []}
probe_results = {'counts': [], 'pyslice': [], 'abtem': [], 'speedups': []}

print("="*50)
print("SYSTEM SIZE SCALING (single probe)")
print(f"PySlice vs AbTem on {device.upper()}")
print("="*50)

for size in supercell_sizes:
    print(f"\n--- Testing {size}³ supercell ---")

    trajectory, atoms = create_trajectory(size)
    box_x, box_y = atoms.cell[0,0], atoms.cell[1,1]
    n_atoms = len(atoms)

    print(f"  {n_atoms} atoms, {box_x:.1f} Å box")

    # Clear cache
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # PySlice
    print("  PySlice...")

    # Prepare data outside timer
    positions = trajectory.positions[0]
    atom_types = trajectory.atom_types
    atom_type_names = [29] * len(atom_types)  # Cu atomic number

    # Calculate grid from box size
    dx = dy = 0.1  # sampling
    dz = 0.5  # slice_thickness
    xs = np.arange(0, box_x, dx)
    ys = np.arange(0, box_y, dy)
    zs = np.arange(0, trajectory.box_matrix[2,2], dz)
    probe_pos = [(box_x/2, box_y/2)]

    # Warmup run (build everything to compile kernels)
    potential_warmup = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
    probe_warmup = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
    batched_probe_warmup = create_batched_probes(probe_warmup, probe_pos, device=device)
    _ = Propagate(batched_probe_warmup, potential_warmup, device=device, progress=False, onthefly=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    del potential_warmup, probe_warmup, batched_probe_warmup

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Timed run - build everything inside timer
    start = time.perf_counter()
    potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
    probe = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
    batched_probe = create_batched_probes(probe, probe_pos, device=device)
    exit_wave = Propagate(batched_probe, potential, device=device, progress=False, onthefly=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    pyslice_time = end - start
    print(f"    Time: {pyslice_time:.3f}s")
    del potential, probe, batched_probe, exit_wave

    # AbTem
    print("  AbTem...")
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    abtem.config.set({"device": abtem_device})

    # Warmup run
    atoms_copy = atoms.copy()
    potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
    probe.grid.match(potential)
    _ = probe.build().multislice(potential).compute()
    del potential, probe

    # Clear and recreate for timed run
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    atoms_copy = atoms.copy()
    potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
    probe.grid.match(potential)

    start = time.perf_counter()
    result = probe.build().multislice(potential).compute()
    end = time.perf_counter()

    abtem_time = end - start
    print(f"    Time: {abtem_time:.3f}s")
    del potential, probe, result
    gc.collect()

    speedup = abtem_time / pyslice_time
    print(f"  Speedup: {speedup:.2f}×")

    system_results['sizes'].append(size)
    system_results['n_atoms'].append(n_atoms)
    system_results['pyslice'].append(pyslice_time)
    system_results['abtem'].append(abtem_time)
    system_results['speedups'].append(speedup)


print("\n" + "="*50)
print("MULTI-PROBE SCALING (10³ supercell)")
print(f"PySlice vs AbTem on {device.upper()}")
print("="*50)

trajectory, atoms = create_trajectory(10)
box_x, box_y = atoms.cell[0,0], atoms.cell[1,1]
print(f"System: {len(atoms)} atoms, {box_x:.1f} Å box\n")

for n_probes in probe_counts:
    print(f"\n--- Testing {n_probes} probes ---")

    # ===== PySlice Test =====
    print(f"  PySlice {device.upper()}...")
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Prepare data outside timer
    positions = trajectory.positions[0]
    atom_types = trajectory.atom_types
    atom_type_names = [29] * len(atom_types)  # Cu atomic number

    # Calculate grid from box size
    dx = dy = 0.1  # sampling
    dz = 0.5  # slice_thickness
    xs = np.arange(0, box_x, dx)
    ys = np.arange(0, box_y, dy)
    zs = np.arange(0, trajectory.box_matrix[2,2], dz)
    probe_pos = probe_positions(box_x, box_y, n_probes)

    # Warmup run (build everything to compile kernels)
    potential_warmup = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
    probe_warmup = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
    batched_probe_warmup = create_batched_probes(probe_warmup, probe_pos, device=device)
    _ = Propagate(batched_probe_warmup, potential_warmup, device=device, progress=False, onthefly=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    del potential_warmup, probe_warmup, batched_probe_warmup

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Timed run - build everything inside timer
    start = time.perf_counter()
    potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
    probe = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
    batched_probe = create_batched_probes(probe, probe_pos, device=device)
    exit_waves = Propagate(batched_probe, potential, device=device, progress=False, onthefly=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    pyslice_time = end - start
    print(f"    Time: {pyslice_time:.3f}s ({pyslice_time/n_probes:.4f}s/probe)")

    del potential, probe, batched_probe, exit_waves

    # ===== AbTem Test =====
    print(f"  AbTem {device.upper()}...")
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    abtem.config.set({"device": abtem_device})

    # Warmup run
    atoms_copy = atoms.copy()
    potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
    probe.grid.match(potential)

    if n_probes == 1:
        _ = probe.build().multislice(potential).compute()
    else:
        positions = probe_positions(box_x, box_y, n_probes)
        scan_pos = [(x/box_x, y/box_y) for x,y in positions]
        n_side = int(np.ceil(np.sqrt(n_probes)))
        start_xy = [min(p[0] for p in scan_pos), min(p[1] for p in scan_pos)]
        end_xy = [max(p[0] for p in scan_pos), max(p[1] for p in scan_pos)]
        if start_xy[0] == end_xy[0]: end_xy[0] += 0.01
        if start_xy[1] == end_xy[1]: end_xy[1] += 0.01
        scan = abtem.GridScan(start=start_xy, end=end_xy, gpts=[n_side, n_side])
        detector = abtem.FlexibleAnnularDetector()
        _ = probe.scan(potential, scan=scan, detectors=detector, max_batch=n_probes).compute()

    if device == 'cuda':
        torch.cuda.synchronize()
    del potential, probe

    # Clear and recreate for timed run
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    atoms_copy = atoms.copy()
    potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
    probe.grid.match(potential)

    if n_probes == 1:
        # Single probe - direct multislice
        start = time.perf_counter()
        result = probe.build().multislice(potential).compute()
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
    else:
        # Multi-probe - use scan
        positions = probe_positions(box_x, box_y, n_probes)
        scan_pos = [(x/box_x, y/box_y) for x,y in positions]
        n_side = int(np.ceil(np.sqrt(n_probes)))

        start_xy = [min(p[0] for p in scan_pos), min(p[1] for p in scan_pos)]
        end_xy = [max(p[0] for p in scan_pos), max(p[1] for p in scan_pos)]
        if start_xy[0] == end_xy[0]: end_xy[0] += 0.01
        if start_xy[1] == end_xy[1]: end_xy[1] += 0.01

        scan = abtem.GridScan(start=start_xy, end=end_xy, gpts=[n_side, n_side])
        detector = abtem.FlexibleAnnularDetector()

        start = time.perf_counter()
        result = probe.scan(potential, scan=scan, detectors=detector, max_batch=n_probes).compute()
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

    abtem_time = end - start
    print(f"    Time: {abtem_time:.3f}s ({abtem_time/n_probes:.4f}s/probe)")

    del potential, probe, result
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Print speedup
    speedup = abtem_time / pyslice_time
    print(f"  Speedup: {speedup:.2f}×")

    probe_results['counts'].append(n_probes)
    probe_results['pyslice'].append(pyslice_time)
    probe_results['abtem'].append(abtem_time)
    probe_results['speedups'].append(speedup)

    # Stop if getting too slow
    if pyslice_time > 30 or abtem_time > 60:
        print(f"  Stopping at {n_probes} probes")
        break

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