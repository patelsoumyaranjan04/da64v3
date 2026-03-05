

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():

    
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, default="best_model.npy")
    parser.add_argument("-d","--dataset", type=str, default="mnist")

    parser.add_argument("-b","--batch_size", type=int, default=64)

    parser.add_argument("-nhl","--num_layers", type=int, default=2)


    parser.add_argument("-sz","--hidden_size", nargs="+", type=int, default=None)

    parser.add_argument("-a","--activation", type=str,
                        choices=["relu","sigmoid","tanh"],
                        default="relu")

    parser.add_argument("-l","--loss", type=str,
                        choices=["cross_entropy","mse"],
                        default="cross_entropy")

    parser.add_argument("-o","--optimizer", type=str, default="sgd")

    parser.add_argument("-lr","--learning_rate", type=float, default=0.001)

    parser.add_argument("-w_i","--weight_init", type=str, default="xavier")

    parser.add_argument("-wd","--weight_decay", type=float, default=0.0,required=False)

    parser.add_argument("-w_p","--wandb_project", type=str, default="ann_assignment",required=False)
    
    args = parser.parse_args()

    if args.hidden_size is None:
        args.hidden_size = [64] * args.num_layers

    return args


def load_model(model_path):
    return np.load(model_path, allow_pickle=True).item()



def evaluate_model(model, X_test, y_test):

    print("X_test shape:", X_test.shape)
    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    return {
        "logits": logits,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():


    args = parse_arguments()

    _, _, X_test, _, _, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)

    model_data = load_model(args.model_path)

    args.num_layers = model_data["num_layers"]
    args.hidden_size = model_data["hidden_size"]

    model = NeuralNetwork(args)
    model.set_weights(model_data["weights"])

    results = evaluate_model(model, X_test, y_test)

    print("Evaluation complete!")
    print(results)


if __name__ == "__main__":
    main()


