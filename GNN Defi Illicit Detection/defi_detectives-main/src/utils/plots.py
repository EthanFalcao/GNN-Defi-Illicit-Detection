import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

def plot_f1(licit_f1_scores, illicit_fl_scores, filename="f1_comparison"):
    # Plots F1 over epochs
    epochs = range(len(illicit_fl_scores))
    plt.plot(epochs, illicit_fl_scores, label='Illicit')
    plt.plot(epochs, licit_f1_scores, label='Licit')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.title("F1 Score per Class", fontsize=16)
    plt.savefig(filename+'.png')
    plt.show()

# Designed for the minority class
def plot_precision_recall(targets, probs, filename="pr_curve"):
    # Claude helped me get the right inputs for precision_recall_curve
    illicit_probs = probs[:, 1]

    illicit_precision, illicit_recall, _ = precision_recall_curve(targets, illicit_probs)
    plt.plot(illicit_recall, illicit_precision, label="Illicit PR Curve")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.title("Precision-Recall Curve", fontsize=16)
    plt.savefig(filename+'.png')
    plt.show()

def plot_loss(loss, filename="focal_loss"):
    epochs = range(len(loss))
    plt.plot(epochs, loss, label='Loss')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Focal Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.title("Focal Loss per Epoch", fontsize=16)
    plt.savefig(filename +'.png')
    plt.show()