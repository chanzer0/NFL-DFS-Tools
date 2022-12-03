import sys
from nfl_showdown_optimizer import *
from nfl_optimizer import *


def main(arguments):
    if len(arguments) < 3 or len(arguments) > 7:
        print('Incorrect usage. Please see `README.md` for proper usage.')
        exit()

    site = arguments[1]
    process = arguments[2]

    if process == 'opto':
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = NFL_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    elif process == 'sd':
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()


if __name__ == "__main__":
    main(sys.argv)
