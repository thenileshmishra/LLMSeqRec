import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(log_file1, log_file2, label1, label2, metric, ylabel, title, save_path):
    df1 = pd.read_csv(log_file1, names=["epoch", metric])
    df2 = pd.read_csv(log_file2, names=["epoch", metric])
    
    plt.figure(figsize=(8, 5))
    plt.plot(df1["epoch"], df1[metric], label=label1, marker='o')
    plt.plot(df2["epoch"], df2[metric], label=label2, marker='x')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def run_all_plots():
    plot_metric("logs/llmseqrec_train_log.csv", "logs/sasrec_train_log.csv",
                "LLMSeqRec", "SASRec", "loss", "Loss", "Training Loss Comparison",
                "logs/loss_plot.png")

    plot_metric("logs/llmseqrec_metrics.csv", "logs/sasrec_metrics.csv",
                "LLMSeqRec", "SASRec", "hit_at_10", "Hit@10", "Hit@10 Comparison",
                "logs/hit_at_10_plot.png")

    plot_metric("logs/llmseqrec_metrics.csv", "logs/sasrec_metrics.csv",
                "LLMSeqRec", "SASRec", "ndcg_at_10", "NDCG@10", "NDCG@10 Comparison",
                "logs/ndcg_at_10_plot.png")

if __name__ == "__main__":
    run_all_plots()
