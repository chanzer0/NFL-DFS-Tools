import sys
from windows_inhibitor import *
from nfl_showdown_optimizer import *
from nfl_optimizer import *
import time


def main(arguments):
    if len(arguments) < 3 or len(arguments) > 7:
        print("Incorrect usage. Please see `README.md` for proper usage.")
        exit()

    site = arguments[1]
    process = arguments[2]

    if process == "opto":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        start = time.time()
        opto = NFL_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()
        end = time.time()
        elapsed = end - start
        minutes, seconds = divmod(elapsed, 60)
        print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds")

    elif process == "sd_opto":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    elif process == "sd_sim":
        import nfl_showdown_simulator

        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        match_lineup_input_to_field_size = True
        if arguments[3] == "cid":
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == "file":
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]
        # if 'match' in arguments:
        #    match_lineup_input_to_field_size = True
        sim = nfl_showdown_simulator.NFL_Showdown_Simulator(
            site, field_size, num_iterations, use_contest_data, use_file_upload
        )
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.save_results()

    elif process == "sim":
        import nfl_gpp_simulator

        site = arguments[1]
        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        match_lineup_input_to_field_size = True
        if arguments[3] == "cid":
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == "file":
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]
        # if 'match' in arguments:
        #    match_lineup_input_to_field_size = True
        sim = nfl_gpp_simulator.NFL_GPP_Simulator(
            site, field_size, num_iterations, use_contest_data, use_file_upload
        )
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()


if __name__ == "__main__":
    main(sys.argv)
