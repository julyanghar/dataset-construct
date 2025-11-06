import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# 颜色：ColorBrewer Set2 (8色)
colors = [
    "#8DA0CB",  # soft periwinkle
    "#A6D854",  # soft spring green
    "#FFD92F",  # pastel yellow
    "#E78AC3",  # soft pink purple
    "#66C2A5",  # mint cyan green
    "#FC8D62",  # peach orange
    "#B3B3B3",  # neutral gray
    "#C7C7E2",  # pale lavender
]

models = [
    "Baseline", "LLava-RLHF", "POVID", "CSR(3Iter)",
    "SIMA", "mDPO", "RE-ALIGN", "Ma-DPO"
]

# 从表格提取数据
pope_r = [88.14, 85.06, 88.31, 87.63, 88.06, 88.10, 88.63, 88.95]
pope_p = [87.23, 84.60, 87.16, 87.00, 87.10, 87.13, 87.43, 87.63]
pope_a = [85.10, 83.40, 85.06, 85.00, 85.03, 85.06, 85.10, 85.40]

hall_q = [10.3297, 10.2859, 10.5495, 10.1099, 10.9890, 9.8901, 11.2088, 11.2088]
hall_f = [18.2081, 18.2081, 18.2081, 18.7861, 17.6301, 18.4971, 18.7861, 19.0751]
hall_easy = [41.7582, 38.2418, 40.6341, 42.9451, 41.7582, 41.8746, 45.5243, 44.9765]
hall_hard = [40.2326, 40.6744, 40.0000, 40.6977, 40.2326, 40.6744, 41.6279, 41.9341]

mm_attr = [3.70, 3.72, 3.67, 3.75, 3.64, 3.70, 3.72, 3.92]
mm_comp = [2.42, 2.63, 2.57, 2.64, 2.69, 2.48, 2.79, 2.92]
mm_rel  = [2.41, 2.48, 2.42, 2.60, 2.59, 2.54, 2.65, 2.75]

sqa =      [66.02, 63.11, 65.98, 65.46, 65.83, 67.53, 68.10, 69.01]
textvqa =  [58.18, 57.46, 58.18, 57.86, 58.48, 57.90, 57.49, 58.91]
viswiz =   [50.03, 49.57, 49.80, 47.02, 49.58, 50.59, 50.13, 50.79]
llavabench=[64.1,  60.2,  67.3,  68.3,  66.9,  59.0,  66.5,  68.1]

def normalize_rows_minmax(data):
    arr = np.array(data, dtype=float)
    minv = arr.min(axis=1, keepdims=True)
    maxv = arr.max(axis=1, keepdims=True)
    return (arr - minv) / (maxv - minv + 1e-12)


# Normalize
data = normalize_rows_minmax([
    pope_r, pope_p, pope_a, hall_q, hall_f, hall_easy, hall_hard, mm_attr, mm_comp, mm_rel, sqa, textvqa, viswiz, llavabench
]).T  # shape: 8 models × 8 metrics


def radar(models, categories, data, title, colors, outfile):
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=13)
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    # 将文字向外移动 (0.05 可以调，比如 0.08 会更明显)
        label.set_position((angle, 0.000001))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.6)

    for i in range(len(models)):
        values = list(data[i]) + [data[i][0]]
        ax.plot(angles, values, linewidth=1.2, color=colors[i], label=models[i])
        ax.fill(angles, values, alpha=0.06, color=colors[i])

    # plt.title(title, fontsize=12, pad=20)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),   # 向下移
        ncol=4,
        frameon=False,
        borderaxespad=1.5              # 与圆保持距离
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    print("Saved:", outfile)
    plt.close()


categories = ["POPE$^r$", "POPE$^p$", "POPE$^a$", "Hallusion$^q$", 
            "Hallusion$^f$", "Hallusion$^{Easy}$", "Hallusion$^{Hard}$",
            "MMHal$^{attr}$", "MMHal$^{com}$", "MMHal$^{rel}$",
            "SQA", "TextVQA", "VisWiz", "LLaVABench"]



# --- Group 1: 前4 + 最后1 ---
models_g1 = models[:4] + models[-1:]
colors_g1 = colors[:4] + colors[-1:]
data_g1 = np.vstack([data[0:4], data[-1]])

radar(
    models_g1, categories, data_g1,
    "Hallucination & POPE (Group 1)",
    colors_g1, "radar_group1.pdf"
)

# --- Group 2: 后4 ---
models_g2 = models[4:8]
colors_g2 = colors[4:8]
data_g2 = data[4:8]

radar(
    models_g2, categories, data_g2,
    "Hallucination & POPE (Group 2)",
    colors_g2, "radar_group2.pdf"
)