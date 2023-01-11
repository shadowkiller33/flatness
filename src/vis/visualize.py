import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

df = pd.read_csv("result.csv")
# mi = df[df["method"] == "MI"]


plot_mi = False
plot_dataset = False
if plot_mi:
    mi = df[(df["method"] == "MI") | (df["method"] == "MI+Flat")]
    mi_rate = mi[mi["metric"] == "Rate"]
    mi_rate_base = mi_rate[mi_rate["model"] == "GPT2-base"]
    mi_rate_ag = mi_rate[mi_rate["dataset"] == "AGNews"]
    if plot_dataset:
        ax = sns.barplot(data=mi_rate_base, x="dataset", y="rate", hue="method")
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(loc="center right", title="Metric")
        plt.title("GPT-base's Rate Performance (MI v.s. MI+Flat)")
        plt.savefig("mi.pdf", dpi=1000)
    else:
        ax = sns.barplot(data=mi_rate_ag, x="model", y="rate", hue="method")
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(loc="center right", title="Metric")
        plt.title("Different GPT model's Rate Performance (MI v.s. MI+Flat)")
        plt.savefig("mi_gpt.pdf", dpi=1000)
else:
    sen = df[(df["method"] == "Sen") | (df["method"] == "Sen+Flat")]
    sen_rate = sen[sen["metric"] == "Rate"]
    sen_rate_base = sen_rate[sen_rate["model"] == "GPT2-base"]
    sen_rate_ag = sen_rate[sen_rate["dataset"] == "AGNews"]
    if plot_dataset:
        ax = sns.barplot(data=sen_rate_base, x="dataset", y="rate", hue="method")
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(loc="center right", title="Metric")
        plt.title("GPT-base's Rate Performance (Sen v.s. Sen+Flat)")
        plt.savefig("sen.pdf", dpi=1000)
    else:
        ax = sns.barplot(data=sen_rate_ag, x="model", y="rate", hue="method")
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(loc="center right", title="Metric")
        plt.title("Different GPT model's Rate Performance (Sen v.s. Sen+Flat)")
        plt.savefig("sen_gpt.pdf", dpi=1000)
