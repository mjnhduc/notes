import argparse
from training.train import train_model
from config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)