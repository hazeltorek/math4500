import matplotlib.pyplot as plt
import numpy as np

# load depth results from results.csv ---NEEDS HEADER TO WORK---
# Expected format: topology,strict,o2  
o2_data = {}
try:
    with open("results.csv") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            topo_code = parts[0]
            strict = parts[1]
            try:
                o2 = float(parts[2])
            except ValueError:
                continue  # skip header
            
            # map codes to display names
            topo_names_map = {"L6": "Line", "Y6": "Y", "G6": "Grid"}
            topo = topo_names_map.get(topo_code, topo_code)
            
            # map strict bool to mode name
            mode = "Sequential" if strict ("True", "true", "1") else "Parallel"
            
            o2_data.setdefault((topo, mode), []).append(o2)
except FileNotFoundError:
    print("Warning: results.csv not found")

# depth table
print("=" * 70)
print("o2 (depth) results")
print("=" * 70)
print(f"{'Config':<25} {'N':<5} {'Mean':<8} {'Median':<8} {'Min':<5} {'Max':<5}")
print("-" * 70)
for (topo, mode), vals in o2_data.items():
    label = f"{topo} ({mode})"
    print(f"{label:<25} {len(vals):<5} {np.mean(vals):<8.2f} {np.median(vals):<8.1f} {int(min(vals)):<5} {int(max(vals)):<5}")

# load crosstalk (o3) results from crosstalk_results.csv ---ALSO NEEDS HEADER---
# format topology,strict,o2,ct_before,ct_after 
ct_data = {}
try:
    with open("crosstalk_results.csv") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split(",")
            topo = parts[0]
            strict = parts[1]
            o2 = float(parts[2])
            ct_before = int(parts[3])
            ct_after = int(parts[4])
            key = (topo, strict)
            ct_data.setdefault(key, []).append((o2, ct_before, ct_after))
except FileNotFoundError:
    print("Warning: crosstalk_results.csv not found, skipping crosstalk analysis")

# crosstalk table
if ct_data:
    print("\n" + "=" * 70)
    print("Crosstalk results (depth o2 enforced, parallel routing)")
    print("=" * 70)
    print(f"{'Topology':<12} {'N':<5} {'avg o2':<8} {'avg ct_before':<14} {'avg ct_after':<13} {'reduction %':<12}")
    print("-" * 70)
    for key, rows in ct_data.items():
        o2s = [r[0] for r in rows]
        cts_before = [r[1] for r in rows]
        cts_after = [r[2] for r in rows]
        avg_b = np.mean(cts_before)
        avg_a = np.mean(cts_after)
        reduction_pct = 100 * (avg_b - avg_a) / avg_b if avg_b > 0 else 0
        print(f"{key[0]:<12} {len(rows):<5} {np.mean(o2s):<8.2f} {avg_b:<14.2f} {avg_a:<13.2f} {reduction_pct:<12.1f}")

# generate depth figure based on topology and constraints
if o2_data:
    topos = ["Line", "Y", "Grid"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seq_means = [np.mean(o2_data[(t, "Sequential")]) if (t, "Sequential") in o2_data else 0 for t in topos]
    seq_stds = [np.std(o2_data[(t, "Sequential")]) if (t, "Sequential") in o2_data else 0 for t in topos]
    par_means = [np.mean(o2_data[(t, "Parallel")]) if (t, "Parallel") in o2_data else 0 for t in topos]
    par_stds = [np.std(o2_data[(t, "Parallel")]) if (t, "Parallel") in o2_data else 0 for t in topos]

    x = np.arange(len(topos))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential routing', color='steelblue', capsize=5)
    ax.bar(x + width/2, par_means, width, yerr=par_stds, label='Parallel routing', color='coral', capsize=5)
    ax.set_xlabel('Topology')
    ax.set_ylabel('Mean dummy timesteps ($o_2$)')
    ax.set_title('Average $o_2$ by topology and routing mode')
    ax.set_xticks(x)
    ax.set_xticklabels(topos)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    plot_data = []
    labels = []
    for topo in topos:
        if (topo, "Sequential") in o2_data:
            plot_data.append(o2_data[(topo, "Sequential")])
            labels.append(f"{topo}\nSequential")
        if (topo, "Parallel") in o2_data:
            plot_data.append(o2_data[(topo, "Parallel")])
            labels.append(f"{topo}\nParallel")

    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
    colors = ['steelblue', 'coral'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('$o_2$ (dummy timesteps)')
    ax.set_title('Distribution of $o_2$ across runs')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('o2_comparison.png', dpi=150)
    plt.show()

# generate figure based on crosstalking edge pairs and depth before/after
if ct_data:
    topo_codes = ["L6", "Y6", "G6"]
    topo_names_map = {"L6": "Line", "Y6": "Y", "G6": "Grid"}
    labels = [topo_names_map[t] for t in topo_codes]

    before_means = [np.mean([r[1] for r in ct_data[(t, "False")]]) for t in topo_codes]
    before_stds = [np.std([r[1] for r in ct_data[(t, "False")]]) for t in topo_codes]
    after_means = [np.mean([r[2] for r in ct_data[(t, "False")]]) for t in topo_codes]
    after_stds = [np.std([r[2] for r in ct_data[(t, "False")]]) for t in topo_codes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(topo_codes))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, before_means, width, yerr=before_stds, label='Before $o_3$ minimization', color='coral', capsize=5)
    ax.bar(x + width/2, after_means, width, yerr=after_stds, label='After $o_3$ minimization', color='steelblue', capsize=5)
    ax.set_xlabel('Topology')
    ax.set_ylabel('Average crosstalking edge pairs')
    ax.set_title('Crosstalk before and after $o_3$ minimization\n(depth $o_2$ enforced, parallel routing)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    plot_data = []
    plot_labels = []
    for topo in topo_codes:
        plot_data.append([r[1] for r in ct_data[(topo, "False")]])
        plot_labels.append(f"{topo_names_map[topo]}\nbefore")
        plot_data.append([r[2] for r in ct_data[(topo, "False")]])
        plot_labels.append(f"{topo_names_map[topo]}\nafter")

    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    colors = ['coral', 'steelblue'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Crosstalking edge pairs')
    ax.set_title('Distribution across runs\n(depth $o_2$ enforced, parallel routing)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('crosstalk_comparison.png', dpi=150)
    plt.show()