import argparse
import os
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", type=str,
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        default="rmsprop")

    parser.add_argument("-a", "--activation", type=str,
                        choices=["relu", "sigmoid", "tanh"],
                        default="relu")

    parser.add_argument("-l", "--loss", type=str,
                        choices=["cross_entropy", "mse"],
                        default="cross_entropy")

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier")

    parser.add_argument("-nhl", "--num_layers", type=int, default=2)

    parser.add_argument("-sz", "--hidden_size",
                        nargs="+", type=int, default=None)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-w_p", "--wandb_project",
                        type=str, default="ann_assignment")

    parser.add_argument("--model_save_path", default="best_model.npy")

    args = parser.parse_args()

    # ---------- hidden layer sanity ----------
    if args.hidden_size is None:
        args.hidden_size = [64] * args.num_layers

    elif len(args.hidden_size) < args.num_layers:
        last = args.hidden_size[-1]
        args.hidden_size = args.hidden_size + \
            [last] * (args.num_layers - len(args.hidden_size))

    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    return args

def save_model(model, args):
    save_path = args.model_save_path

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    model_data = {
        "weights": model.get_weights(),
        "architecture": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size
        }
    }

    np.save(save_path, model_data, allow_pickle=True)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(save_path)),
        "config.json"
    )

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Model stored at: {save_path}")
    print(f"Config stored at: {config_path}")


def main():
    args = parse_arguments()

    print(f"Loading dataset: {args.dataset}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)

    print("Starting training...")

    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Evaluating on test set...")

    accuracy = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {accuracy:.4f}")

    save_model(model, args)


if __name__ == "__main__":
    main()











