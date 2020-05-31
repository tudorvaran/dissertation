import json

from matplotlib import pyplot as plt


with open("results_dimensions.json", "r") as f:
    data = json.load(f)

metrics = [metric for metric in list(data.values())[0][0]["mean"]]

for metric in metrics:
    xs = []
    ys = []
    for active_bits in sorted(list(data.keys())):
        if active_bits == "means":
            continue

        for entry in data[active_bits]:
            xs.append(active_bits)
            ys.append(entry["mean"][metric])

    mean_xs = []
    mean_ys = []
    for active_bits in sorted(list(data["means"].keys())):
        mean_xs.append(active_bits)
        mean_ys.append(data["means"][active_bits][metric])

    plt.plot(xs, ys, 'b^')
    plt.plot(mean_xs, mean_ys, 'g-')

    with open(f"dim_graphs_{metric}.png", "wb") as f:
        plt.savefig(f)
    plt.close()

