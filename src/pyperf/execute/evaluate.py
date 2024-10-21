import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create /plots
os.makedirs("plots", exist_ok=True)

# Time data
time1 = """
A:
Execution time: 110.785194s
Execution time: 110.968854s
Execution time: 111.177443s
Execution time: 113.135757s
Execution time: 111.675586s

B:
Execution time: 23.462409s
Execution time: 23.436030s
Execution time: 23.610371s
Execution time: 23.736650s
Execution time: 24.024381s
"""

time2 = """
A:
Execution time: 106.219454s
Execution time: 107.948086s
Execution time: 110.418012s
Execution time: 107.410151s
Execution time: 107.541450s

B:
Execution time: 23.446738s
Execution time: 23.539465s
Execution time: 23.328237s
Execution time: 23.862795s
Execution time: 23.448002s
"""

time3 = """
A:
Execution time: 104.214771s
Execution time: 106.154633s
Execution time: 108.376314s
Execution time: 107.014499s
Execution time: 107.281865s

B:
Execution time: 23.165330s
Execution time: 23.831978s
Execution time: 23.865407s
Execution time: 23.700852s
Execution time: 23.756649s
"""

time4 = """
A:
Execution time: 109.785981s
Execution time: 111.679360s
Execution time: 111.180119s
Execution time: 111.934369s
Execution time: 112.060562s

B:
Execution time: 23.303541s
Execution time: 23.663041s
Execution time: 23.178878s
Execution time: 24.226274s
Execution time: 23.419775s
"""

time5 = """
A:
Execution time: 95.006532s
Execution time: 92.567094s
Execution time: 92.523155s
Execution time: 92.673451s
Execution time: 92.691092s

B:
Execution time: 18.104597s
Execution time: 18.191019s
Execution time: 18.110599s
Execution time: 18.028916s
Execution time: 18.228759s
"""


# Function to parse time strings and convert to seconds
def parse_times(time_str):
    pattern = r"Execution time:\s+([\d\.]+)s"
    times = []
    for line in time_str.strip().split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            seconds = float(match.group(1))
            times.append(seconds)
    return times


# Split times into subgroups (each subgroup has 5 measurements)
def split_subgroups(times):
    commit_a = times[:5]
    commit_b = times[5:10]

    return commit_a, commit_b


# Compute statistics: mean and standard deviation
def compute_stats(times):
    mean = np.mean(times)
    std_dev = np.std(times, ddof=1)  # Sample standard deviation
    return mean, std_dev


# Split times for each machine
commit_a_times1, commit_b_times1 = split_subgroups(parse_times(time1))
commit_a_times2, commit_b_times2 = split_subgroups(parse_times(time2))
commit_a_times3, commit_b_times3 = split_subgroups(parse_times(time3))
commit_a_times4, commit_b_times4 = split_subgroups(parse_times(time4))
commit_a_times5, commit_b_times5 = split_subgroups(parse_times(time5))


# Compute stats for each subgroup
stats = {}
for i, (sbg1, sbg2) in enumerate(
    [
        (commit_a_times1, commit_b_times1),
        (commit_a_times2, commit_b_times2),
        (commit_a_times3, commit_b_times3),
        (commit_a_times4, commit_b_times4),
        (commit_a_times5, commit_b_times5),
    ],
    1,
):
    mean1, std1 = compute_stats(sbg1)
    mean2, std2 = compute_stats(sbg2)
    opt = ((mean1 - mean2) / mean1) * 100  # Percentage change
    combined_times = sbg1 + sbg2
    noise = np.std(combined_times, ddof=1)
    stats[f"M{i}"] = {
        "C1 Mean": mean1,
        "C1 StdDev": std1,
        "C2 Mean": mean2,
        "C2 StdDev": std2,
        "Opt (%)": opt,
        "Noise (StdDev of All)": noise,
    }

# Print statistics
for machine, machine_stats in stats.items():
    print(f"{machine}:")
    for stat_name, value in machine_stats.items():
        print(f"  {stat_name}: {value:.3f}")
    print()

# Plotting
sns.set_style("whitegrid")
machines = ["M1", "M2", "M3", "M4", "M5"]
subgroup_labels = ["Commit A", "Commit B"]

# Collect all times
all_times = [
    commit_a_times1,
    commit_b_times1,
    commit_a_times2,
    commit_b_times2,
    commit_a_times3,
    commit_b_times3,
    commit_a_times4,
    commit_b_times4,
    commit_a_times5,
    commit_b_times5,
]

# Flatten the list for boxplot
flattened_times = [time for subgroup in all_times for time in subgroup]
group_labels = sum(
    [[f"M{idx1} C{idx2}"] * 5 for idx1 in range(1, 6) for idx2 in range(1, 3)], []
)

#################### # Boxplot of times for each subgroup on each machine ####################

# Create DataFrame for seaborn
import pandas as pd

df = pd.DataFrame({"Time (s)": flattened_times, "Commit": group_labels})

plt.figure(figsize=(12, 6))
sns.boxplot(
    x="Commit", y="Time (s)", data=df, palette="Pastel1", hue="Commit", legend=False
)
plt.title("Time Measurements Across Machines and Commits", fontsize=16)
plt.xlabel("Machine and Commit", fontsize=14)
plt.ylabel("Time (seconds)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("./plots/boxplot.png")


#################### Bar plot of mean times with error bars ####################

means = []
stds = []
labels = []
for machine in ["M1", "M2", "M3", "M4", "M5"]:
    means.append(stats[machine]["C1 Mean"])
    stds.append(stats[machine]["C1 StdDev"])
    means.append(stats[machine]["C2 Mean"])
    stds.append(stats[machine]["C2 StdDev"])
    labels.extend([f"{machine} {label}" for label in ["C1", "C2"]])


x = np.arange(len(labels))

plt.figure(figsize=(12, 6))
barlist = plt.bar(x, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
plt.title("Average Time with Error Bars", fontsize=16)
plt.xlabel("Machine and Commit", fontsize=14)
plt.ylabel("Average Time (seconds)", fontsize=14)
plt.xticks(x, labels, fontsize=12)
plt.yticks(fontsize=12)

# Annotate optimization percentages
for i in range(0, len(means), 2):
    opt = stats[f"M{i//2+1}"]["Opt (%)"]
    plt.text(
        i + 1,
        max(means) * 0.3,
        f"{opt:.2f}%",
        ha="center",
        fontsize=12,
        color="green",
    )

plt.tight_layout()
plt.savefig("./plots/barplot.png")

#################### Boxplot of times for each subgroup ####################

commit_a_combined = (
    commit_a_times1
    + commit_a_times2
    + commit_a_times3
    + commit_a_times4
    + commit_a_times5
)
commit_b_combined = (
    commit_b_times1
    + commit_b_times2
    + commit_b_times3
    + commit_b_times4
    + commit_b_times5
)

df = pd.DataFrame(
    {
        "Commit A": commit_a_combined,
        "Commit B": commit_b_combined,
    }
)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, palette="Pastel1")
plt.title("Time Measurements Across Commits", fontsize=16)
plt.xlabel("Commits", fontsize=14)
plt.ylabel("Time (seconds)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("./plots/boxplot_commits.png")

#################### Summary ####################

opts = []
for i, (v1, v2) in enumerate(zip(commit_a_combined, commit_b_combined), 1):
    opt1 = ((v1 - v2) / v1) * 100
    opts.append(opt1)

print("Summary:")
print(
    f"Commit A: {np.mean(commit_a_combined):.3f}s ({np.std(commit_a_combined, ddof=1):.2f})"
)
print(
    f"Commit B: {np.mean(commit_b_combined):.3f}s ({np.std(commit_b_combined, ddof=1):.2f})"
)

print(f"Opt: {np.mean(opts):.2f}% ({np.std(opts, ddof=1):.2f})")
