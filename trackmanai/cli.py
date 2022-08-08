"""
Console script for qaqgplayground.
"""

# Imports
import argparse
import sys
from neat_trainer import train_neat


# Main function
def main():
    """ Creates a parser for AI development / showcase.
    Possible actions : 
        - Train a NEAT model
    """
    # Create parser
    parser = argparse.ArgumentParser()

    # Train NEAT model subparser
    subparser = parser.add_subparsers()
    parser_train_neat = subparser.add_parser('train_neat', help="Trains a NEAT model following a given config.")
    parser_train_neat.add_argument('--run_config', type=str, default="./models/config.ini")
    parser_train_neat.add_argument('--model_config', type=str, default="./models/NEAT/config-feedforward")
    parser_train_neat.add_argument('--checkpoint', type=str, default="./models/NEAT/Checkpoints/checkpoint-0")
    parser_train_neat.add_argument('--no_generations', type=int, default=1000)
    parser_train_neat.set_defaults(func=train_neat)

    # Call function
    try:
        args = parser.parse_args()
        func_to_call = args.func
        parsed_args_dict = {k.replace('-', '_'): v for k, v in vars(args).items() if k != 'func'}
        func_to_call(**parsed_args_dict)
    except Exception as err:
        print(err)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())