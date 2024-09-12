import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import statistics

# import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from numba import jit
import datetime

@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2


def salary_boost(salary, max_salary):
    # Linear boost
    # return salary / max_salary

    # Non-linear boost (can adjust exponent for more/less emphasis)
    return (salary / max_salary) ** 2


class NFL_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    game_info = {}
    id_name_dict = {}
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None
    use_lineup_input = None
    matchups = set()
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    teams_dict = collections.defaultdict(list)  # Initialize teams_dict
    correlation_rules = {}
    seen_lineups = {}
    seen_lineups_ix = {}
    position_map = {
        0: ["DST"],
        1: ["QB"],
        2: ["RB"],
        3: ["RB"],
        4: ["WR"],
        5: ["WR"],
        6: ["WR"],
        7: ["TE"],
        8: ["RB", "WR", "TE"],
    }

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
    ):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)
        self.load_team_stacks()

        # ownership_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["ownership_path"]),
        # )
        # self.load_ownership(ownership_path)

        # boom_bust_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        # )
        # self.load_boom_bust(boom_bust_path)

        #       batting_order_path = os.path.join(
        #           os.path.dirname(__file__),
        #            "../{}_data/{}".format(site, self.config["batting_order_path"]),
        #        )
        #        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 50000

        elif site == "fd":
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 60000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0

        # self.adjust_default_stdev()
        self.assertPlayerDict()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        # if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
        # self.generate_field_lineups()
        self.load_correlation_rules()

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config["pct_field_using_stacks"])
        self.default_qb_var = float(self.config["default_qb_var"])
        self.default_skillpos_var = float(self.config["default_skillpos_var"])
        self.default_def_var = float(self.config["default_def_var"])
        self.overlap_limit = float(self.config["num_players_vs_def"])
        self.pct_field_double_stacks = float(self.config["pct_field_double_stacks"])
        self.correlation_rules = self.config["custom_correlations"]

    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
                )
                self.player_dict.pop(p)

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        # print(s['Name'],s['ID'])
        # print(self.player_dict)
        problem = plp.LpProblem("NFL", plp.LpMaximize)
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["fieldFpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Need 1 quarterback
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 8 per team in case of weird issues with stacking on short slates
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 8
                )

        elif self.site == "fd":
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 4 per team
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 4
                )

        # print(f"Problem Name: {problem.name}")
        # print(f"Sense: {problem.sense}")

        # # Print the objective
        # print("\nObjective:")
        # try:
        #     for v, coef in problem.objective.items():
        #         print(f"{coef}*{v.name}", end=' + ')
        # except Exception as e:
        #     print(f"Error while printing objective: {e}")

        # # Print the constraints
        # print("\nConstraints:")
        # for constraint in problem.constraints.values():
        #     try:
        #         # Extract the left-hand side, right-hand side, and the operator
        #         lhs = "".join(f"{coef}*{var.name}" for var, coef in constraint.items())
        #         rhs = constraint.constant
        #         if constraint.sense == 1:
        #             op = ">="
        #         elif constraint.sense == -1:
        #             op = "<="
        #         else:
        #             op = "="
        #         print(f"{lhs} {op} {rhs}")
        #     except Exception as e:
        #         print(f"Error while printing constraint: {e}")

        # # Print the variables
        # print("\nVariables:")
        # try:
        #     for v in problem.variables():
        #         print(f"{v.name}: LowBound={v.lowBound}, UpBound={v.upBound}, Cat={v.cat}")
        # except Exception as e:
        #     print(f"Error while printing variable: {e}")
        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )
        except TypeError:
            for p, s in self.player_dict.items():
                if s["ID"] == 0:
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] == "":
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] is None:
                    print(s["Name"])
        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(
            r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)",
            game_string,
        )

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                player_name = row[name_key].replace("-", "#").lower().strip()
                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                        player_name = row['last name'].replace("-", "#").lower().strip()
                # if qb and dst not in position add flex
                if "QB" not in position and "DST" not in position:
                    position.append("FLEX")
                team_key = "teamabbrev" if self.site == "dk" else "team"
                team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                game_info_str = row["game info"] if self.site == "dk" else row["game"]
                result = self.extract_matchup_time(game_info_str)
                match = re.search(pattern="(\w{2,4}@\w{2,4})", string=row[game_info])
                if match:
                    opp = match.groups()[0].split("@")
                    self.matchups.add((opp[0], opp[1]))
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                if result:
                    matchup, game_time = result
                    self.game_info[opp] = game_time
                # if not opp:
                #    print(row)
                pos_str = str(position)
                if (player_name, pos_str, team) in self.player_dict:
                    self.player_dict[(player_name, pos_str, team)]["ID"] = str(
                        row["id"]
                    )
                    self.player_dict[(player_name, pos_str, team)]["Team"] = row[
                        team_key
                    ]
                    self.player_dict[(player_name, pos_str, team)]["Opp"] = team_opp
                    self.player_dict[(player_name, pos_str, team)]["Matchup"] = opp
                self.id_name_dict[str(row["id"])] = row[name_key]

    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["field size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["entry fee"])
                # multi-position payouts
                if "-" in row["place"]:
                    indices = row["place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for primary_player in self.correlation_rules.keys():
                # Convert primary_player to the consistent format
                formatted_primary_player = (
                    primary_player.replace("-", "#").lower().strip()
                )
                for (
                    player_name,
                    pos_str,
                    team,
                ), player_data in self.player_dict.items():
                    if formatted_primary_player == player_name:
                        for second_entity, correlation_value in self.correlation_rules[
                            primary_player
                        ].items():
                            # Convert second_entity to the consistent format
                            formatted_second_entity = (
                                second_entity.replace("-", "#").lower().strip()
                            )

                            # Check if the formatted_second_entity is a player name
                            found_second_entity = False
                            for (
                                se_name,
                                se_pos_str,
                                se_team,
                            ), se_data in self.player_dict.items():
                                if formatted_second_entity == se_name:
                                    player_data["Player Correlations"][
                                        formatted_second_entity
                                    ] = correlation_value
                                    se_data["Player Correlations"][
                                        formatted_primary_player
                                    ] = correlation_value
                                    found_second_entity = True
                                    break

                            # If the second_entity is not found as a player, assume it's a position and update 'Correlations'
                            if not found_second_entity:
                                player_data["Correlations"][
                                    second_entity
                                ] = correlation_value

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower().strip()
                try:
                    fpts = float(row["fpts"])
                except:
                    fpts = 0
                    print(
                        "unable to load player fpts: "
                        + player_name
                        + ", fpts:"
                        + row["fpts"]
                    )
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                # if qb and dst not in position add flex
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                if "QB" not in position and "DST" not in position:
                    position.append("FLEX")
                pos = position[0]
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                        if position == "QB":
                            stddev = fpts * self.default_qb_var
                        elif position == "DST":
                            stddev = fpts * self.default_def_var
                        else:
                            stddev = fpts * self.default_skillpos_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    if position == "QB":
                        stddev = fpts * self.default_qb_var
                    elif position == "DST":
                        stddev = fpts * self.default_def_var
                    else:
                        stddev = fpts * self.default_skillpos_var
                # check if ceiling exists in row columns
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if pos == "QB":
                    corr = {
                        "QB": 1,
                        "RB": 0.08,
                        "WR": 0.62,
                        "TE": 0.32,
                        "DST": -0.09,
                        "Opp QB": 0.24,
                        "Opp RB": 0.04,
                        "Opp WR": 0.19,
                        "Opp TE": 0.1,
                        "Opp DST": -0.41,
                    }
                elif pos == "RB":
                    corr = {
                        "QB": 0.08,
                        "RB": 1,
                        "WR": -0.09,
                        "TE": -0.02,
                        "DST": 0.07,
                        "Opp QB": 0.04,
                        "Opp RB": -0.08,
                        "Opp WR": 0.01,
                        "Opp TE": 0.03,
                        "Opp DST": -0.33,
                    }
                elif pos == "WR":
                    corr = {
                        "QB": 0.62,
                        "RB": -0.09,
                        "WR": 1,
                        "TE": -0.07,
                        "DST": -0.08,
                        "Opp QB": 0.19,
                        "Opp RB": 0.01,
                        "Opp WR": 0.16,
                        "Opp TE": 0.08,
                        "Opp DST": -0.22,
                    }
                elif pos == "TE":
                    corr = {
                        "QB": 0.32,
                        "RB": -0.02,
                        "WR": -0.07,
                        "TE": 1,
                        "DST": -0.08,
                        "Opp QB": 0.1,
                        "Opp RB": 0.03,
                        "Opp WR": 0.08,
                        "Opp TE": 0,
                        "Opp DST": -0.14,
                    }
                elif pos == "DST":
                    corr = {
                        "QB": -0.09,
                        "RB": 0.07,
                        "WR": -0.08,
                        "TE": -0.08,
                        "DST": 1,
                        "Opp QB": -0.41,
                        "Opp RB": -0.33,
                        "Opp WR": -0.22,
                        "Opp TE": -0.14,
                        "Opp DST": -0.27,
                    }
                team = row["team"]
                if team == "LA":
                    team = "LAR"
                if self.site == "fd":
                    if team == "JAX":
                        team = "JAC"
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                }

                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")

                self.player_dict[(player_name, pos_str, team)] = player_data
                self.teams_dict[team].append(
                    player_data
                )  # Add player data to their respective team

    def load_team_stacks(self):
        # Initialize a dictionary to hold QB ownership by team
        qb_ownership_by_team = {}

        for p in self.player_dict:
            # Check if player is a QB
            if "QB" in self.player_dict[p]["Position"]:
                # Fetch the team of the QB
                team = self.player_dict[p]["Team"]

                # Convert the ownership percentage string to a float and divide by 100
                own_percentage = float(self.player_dict[p]["Ownership"]) / 100

                # Add the ownership to the accumulated ownership for the team
                if team in qb_ownership_by_team:
                    qb_ownership_by_team[team] += own_percentage
                else:
                    qb_ownership_by_team[team] = own_percentage

        # Now, update the stacks_dict with the QB ownership by team
        for team, own_percentage in qb_ownership_by_team.items():
            self.stacks_dict[team] = own_percentage

    def extract_id(self, cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[1]
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            j = 0
            for i, row in reader.iterrows():
                # print(row)
                if i == self.field_size:
                    break
                lineup = [self.extract_id(str(row[j])) for j in range(9)]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                if not error:
                    # reshuffle lineup to match temp_roster_construction
                    temp_roster_construction = [
                        "DST",
                        "QB",
                        "RB",
                        "RB",
                        "WR",
                        "WR",
                        "WR",
                        "TE",
                        "FLEX",
                    ]
                    shuffled_lu = []

                    id_to_player_dict = {
                        v["ID"]: v for k, v in self.player_dict.items()
                    }
                    lineup_copy = lineup.copy()
                    position_counts = {
                        "DST": 0,
                        "QB": 0,
                        "RB": 0,
                        "WR": 0,
                        "TE": 0,
                        "FLEX": 0,
                    }
                    z = 0

                    while z < 9:
                        for t in temp_roster_construction:
                            if position_counts[t] < temp_roster_construction.count(t):
                                for l in lineup_copy:
                                    player_info = id_to_player_dict.get(l)
                                    if player_info and t in player_info["Position"]:
                                        shuffled_lu.append(l)
                                        lineup_copy.remove(l)
                                        position_counts[t] += 1
                                        z += 1
                                        if z == 9:
                                            break
                            if z == 9:
                                break
                    lineup_list = sorted(shuffled_lu)           
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j
                        self.field_lineups[j] = {
                            "Lineup": shuffled_lu,
                            "Wins": 0,
                            "Top1Percent": 0,
                            "ROI": 0,
                            "Cashes": 0,
                            "Type": "opto",
                            "Count" : 1
                        }
                        j += 1
        print("loaded {} lineups".format(j))
        #print(len(self.field_lineups))

    @staticmethod
    def generate_lineups(
        lu_num,
        ids,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        teams,
        opponents,
        team_stack,
        stack_len,
        overlap_limit,
        max_stack_len,
        matchups,
        num_players_in_roster,
        site,
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        iteration_count = 0
        total_players = num_players_in_roster
        issue = ""
        complete = ""
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)
        reasonable_stack_projection = optimal_score - (
            (max_pct_off_optimal * 1.25) * optimal_score
        )
        max_players_per_team = 4 if site == "fd" else None
        # reject_counters = {
        #     "salary_too_low": 0,
        #     "salary_too_high": 0,
        #     "projection_too_low": 0,
        #     "invalid_matchups": 0,
        #     "stack_length_insufficient": 0,
        # }
        # print(lu_num, ' started',  team_stack, max_stack_len)
        while reject:
            iteration_count += 1
            if team_stack == "":
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                lineup = []
                player_teams = []
                def_opps = []
                players_opposing_def = 0
                lineup_matchups = []
                k = 0
                for pos in pos_matrix.T:
                    if k < 1:
                        # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                        valid_players = np.nonzero((pos > 0) & (in_lineup == 0))[0]
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = ownership[valid_players]
                        prob_list = prob_list / prob_list.sum()
                        try:
                            choice = rng.choice(plyr_list, p=prob_list)
                        except:
                            print(plyr_list, prob_list)
                            print("find failed on nonstack and first player selection")
                        choice_idx = np.nonzero(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += salaries[choice_idx]
                        proj += projections[choice_idx]
                        def_opp = opponents[choice_idx][0]
                        lineup_matchups.append(matchups[choice_idx[0]])
                        player_teams.append(teams[choice_idx][0])
                    if k >= 1:
                        remaining_salary = salary_ceiling - salary
                        if players_opposing_def < overlap_limit:
                            if k == total_players - 1:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (salary + salaries >= salary_floor)
                                )[0]
                            else:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            if k == total_players - 1:
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                            try:
                                if k == total_players - 1:
                                    choice = rng.choice(
                                        plyr_list, p=boosted_probabilities
                                    )
                                else:
                                    choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                                salary = 0
                                proj = 0
                                if team_stack == "":
                                    lineup = []
                                else:
                                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(
                                        str
                                    )
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                            choice_idx = np.nonzero(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            player_teams.append(teams[choice_idx][0])
                            lineup_matchups.append(matchups[choice_idx[0]])
                            if teams[choice_idx][0] == def_opp:
                                players_opposing_def += 1
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if any(
                                    count > max_players_per_team
                                    for count in team_count.values()
                                ):
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                        else:
                            if k == total_players - 1:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (salary + salaries >= salary_floor)
                                    & (teams != def_opp)
                                )[0]
                            else:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (teams != def_opp)
                                )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            boosted_salaries = np.array(
                                [
                                    salary_boost(s, salary_ceiling)
                                    for s in salaries[valid_players]
                                ]
                            )
                            boosted_probabilities = prob_list * boosted_salaries
                            boosted_probabilities /= (
                                boosted_probabilities.sum()
                            )  # normalize to ensure it sums to 1
                            try:
                                choice = rng.choice(plyr_list, p=boosted_probabilities)
                            except:
                                salary = 0
                                proj = 0
                                if team_stack == "":
                                    lineup = []
                                else:
                                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(
                                        str
                                    )
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                            choice_idx = np.nonzero(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            player_teams.append(teams[choice_idx][0])
                            lineup_matchups.append(matchups[choice_idx[0]])
                            if teams[choice_idx][0] == def_opp:
                                players_opposing_def += 1
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if any(
                                    count > max_players_per_team
                                    for count in team_count.values()
                                ):
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                    k += 1
                # Must have a reasonable salary
                # if salary > salary_ceiling:
                #     reject_counters["salary_too_high"] += 1
                # elif salary < salary_floor:
                #     reject_counters["salary_too_low"] += 1
                if salary >= salary_floor and salary <= salary_ceiling:
                    # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                    if proj >= reasonable_projection:
                        if len(set(lineup_matchups)) > 1:
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if all(
                                    count <= max_players_per_team
                                    for count in team_count.values()
                                ):
                                    reject = False
                                    lus[lu_num] = {
                                        "Lineup": lineup,
                                        "Wins": 0,
                                        "Top1Percent": 0,
                                        "ROI": 0,
                                        "Cashes": 0,
                                        "Type": "generated",
                                        "Count": 0
                                    }
                                    if len(set(lineup)) != 9:
                                        print(
                                            "non stack lineup dupes",
                                            lu_num,
                                            plyr_stack_indices,
                                            str(lu_num),
                                            salaries[plyr_stack_indices],
                                            lineup,
                                            stack_len,
                                            team_stack,
                                            x,
                                        )
                            else:
                                reject = False
                                lus[lu_num] = {
                                    "Lineup": lineup,
                                    "Wins": 0,
                                    "Top1Percent": 0,
                                    "ROI": 0,
                                    "Cashes": 0,
                                    "Type": "generated",
                                    "Count": 0
                                }
                                if len(set(lineup)) != 9:
                                    print(
                                        "stack lineup dupes",
                                        lu_num,
                                        plyr_stack_indices,
                                        str(lu_num),
                                        salaries[plyr_stack_indices],
                                        lineup,
                                        stack_len,
                                        team_stack,
                                        x,
                                    )
                            # complete = 'completed'
                            # print(str(lu_num) + ' ' + complete)
                    #     else:
                    #         reject_counters["invalid_matchups"] += 1
                    # else:
                    #     reject_counters["projection_too_low"] += 1
            else:
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                player_teams = []
                def_opps = []
                lineup_matchups = []
                filled_pos = np.zeros(shape=pos_matrix.shape[1])
                team_stack_len = 0
                k = 0
                stack = True
                lineup = np.zeros(shape=pos_matrix.shape[1]).astype(str)
                valid_team = np.nonzero(teams == team_stack)[0]
                # select qb
                qb = np.unique(
                    valid_team[np.nonzero(pos_matrix[valid_team, 1] > 0)[0]]
                )[0]
                salary += salaries[qb]
                proj += projections[qb]
                # print(salary)
                team_stack_len += 1
                lineup[1] = ids[qb]
                in_lineup[qb] = 1
                lineup_matchups.append(matchups[qb])
                valid_players = np.unique(
                    valid_team[np.nonzero(pos_matrix[valid_team, 4:8] > 0)[0]]
                )
                player_teams.append(teams[qb])
                players_opposing_def = 0
                plyr_list = ids[valid_players]
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()
                while stack:
                    try:
                        choices = rng.choice(
                            a=plyr_list, p=prob_list, size=stack_len, replace=False
                        )
                        if len(set(choices)) != len(choices):
                            print(
                                "choice dupe",
                                plyr_stack_indices,
                                str(lu_num),
                                salaries[plyr_stack_indices],
                                lineup,
                                stack_len,
                                team_stack,
                                x,
                            )
                    except:
                        stack = False
                        continue
                    plyr_stack_indices = np.nonzero(np.in1d(ids, choices))[0]
                    x = 0
                    for p in plyr_stack_indices:
                        player_placed = False
                        for l in np.nonzero(pos_matrix[p] > 0)[0]:
                            if lineup[l] == "0.0":
                                lineup[l] = ids[p]
                                lineup_matchups.append(matchups[p])
                                player_teams.append(teams[p])
                                x += 1
                                player_placed = True
                                break
                            if player_placed:
                                break
                    # print(plyr_stack_indices, str(lu_num), salaries[plyr_stack_indices], lineup, stack_len, x)
                    if x == stack_len:
                        in_lineup[plyr_stack_indices] = 1
                        salary += sum(salaries[plyr_stack_indices])
                        # rint(salary)
                        proj += sum(projections[plyr_stack_indices])
                        # print(proj)
                        team_stack_len += stack_len
                        x = 0
                        stack = False
                    else:
                        stack = False
                # print(sum(in_lineup), stack_len)
                for ix, (l, pos) in enumerate(zip(lineup, pos_matrix.T)):
                    if l == "0.0":
                        if k < 1:
                            valid_players = np.nonzero(
                                (pos > 0) & (in_lineup == 0) & (opponents != team_stack)
                            )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being selected based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            try:
                                choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                print("find failed on stack and first player selection")

                            #    print(k, pos)
                            choice_idx = np.nonzero(ids == choice)[0]
                            in_lineup[choice_idx] = 1
                            try:
                                lineup[ix] = str(choice)
                            except IndexError:
                                print(lineup, choice, ix)
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            def_opp = opponents[choice_idx][0]
                            lineup_matchups.append(matchups[choice_idx[0]])
                            player_teams.append(teams[choice_idx][0])
                            k += 1
                        elif k >= 1:
                            remaining_salary = salary_ceiling - salary
                            if players_opposing_def < overlap_limit:
                                if k == total_players - 1:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (salary + salaries >= salary_floor)
                                    )[0]
                                else:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                    )[0]
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                if k == total_players - 1:
                                    boosted_salaries = np.array(
                                        [
                                            salary_boost(s, salary_ceiling)
                                            for s in salaries[valid_players]
                                        ]
                                    )
                                    boosted_probabilities = prob_list * boosted_salaries
                                    boosted_probabilities /= (
                                        boosted_probabilities.sum()
                                    )  # normalize to ensure it sums to 1
                                try:
                                    if k == total_players - 1:
                                        choice = rng.choice(
                                            plyr_list, p=boosted_probabilities
                                        )
                                    else:
                                        choice = rng.choice(plyr_list, p=prob_list)
                                except:
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                choice_idx = np.nonzero(ids == choice)[0]
                                try:
                                    lineup[ix] = str(choice)
                                except IndexError:
                                    print(lineup, choice, ix)
                                in_lineup[choice_idx] = 1
                                salary += salaries[choice_idx]
                                proj += projections[choice_idx]
                                player_teams.append(teams[choice_idx][0])
                                lineup_matchups.append(matchups[choice_idx[0]])
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if any(
                                        count > max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        salary = 0
                                        proj = 0
                                        if team_stack == "":
                                            lineup = []
                                        else:
                                            lineup = np.zeros(
                                                shape=pos_matrix.shape[1]
                                            ).astype(str)
                                        player_teams = []
                                        def_opps = []
                                        players_opposing_def = 0
                                        lineup_matchups = []
                                        in_lineup.fill(0)  # Reset the in_lineup array
                                        k = 0  # Reset the player index
                                        continue  # Skip to the next iteration of the while loop
                                if teams[choice_idx][0] == def_opp:
                                    players_opposing_def += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                            else:
                                if k == total_players - 1:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (salary + salaries >= salary_floor)
                                        & (teams != def_opp)
                                    )[0]
                                else:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (teams != def_opp)
                                    )[0]
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                                try:
                                    choice = rng.choice(
                                        plyr_list, p=boosted_probabilities
                                    )
                                except:
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                choice_idx = np.nonzero(ids == choice)[0]
                                lineup[ix] = str(choice)
                                in_lineup[choice_idx] = 1
                                salary += salaries[choice_idx]
                                proj += projections[choice_idx]
                                player_teams.append(teams[choice_idx][0])
                                lineup_matchups.append(matchups[choice_idx[0]])
                                if teams[choice_idx][0] == def_opp:
                                    players_opposing_def += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if any(
                                        count > max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        salary = 0
                                        proj = 0
                                        if team_stack == "":
                                            lineup = []
                                        else:
                                            lineup = np.zeros(
                                                shape=pos_matrix.shape[1]
                                            ).astype(str)
                                        player_teams = []
                                        def_opps = []
                                        players_opposing_def = 0
                                        lineup_matchups = []
                                        in_lineup.fill(0)  # Reset the in_lineup array
                                        k = 0  # Reset the player index
                                        continue  # Skip to the next iteration of the while loop
                            k += 1
                    else:
                        k += 1
                # Must have a reasonable salary
                if team_stack_len >= stack_len:
                    if salary >= salary_floor and salary <= salary_ceiling:
                        # loosening reasonable projection constraint for team stacks
                        if proj >= reasonable_stack_projection:
                            if len(set(lineup_matchups)) > 1:
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if all(
                                        count <= max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        reject = False
                                        lus[lu_num] = {
                                            "Lineup": lineup,
                                            "Wins": 0,
                                            "Top1Percent": 0,
                                            "ROI": 0,
                                            "Cashes": 0,
                                            "Type": "generated",
                                            "Count": 0,
                                        }
                                        if len(set(lineup)) != 9:
                                            print(
                                                "stack lineup dupes",
                                                lu_num,
                                                plyr_stack_indices,
                                                str(lu_num),
                                                salaries[plyr_stack_indices],
                                                lineup,
                                                stack_len,
                                                team_stack,
                                                x,
                                            )

                                else:
                                    reject = False
                                    lus[lu_num] = {
                                        "Lineup": lineup,
                                        "Wins": 0,
                                        "Top1Percent": 0,
                                        "ROI": 0,
                                        "Cashes": 0,
                                        "Type": "generated",
                                        "Count": 0,
                                    }
                                    if len(set(lineup)) != 9:
                                        print(
                                            "stack lineup dupes",
                                            lu_num,
                                            plyr_stack_indices,
                                            str(lu_num),
                                            salaries[plyr_stack_indices],
                                            lineup,
                                            stack_len,
                                            team_stack,
                                            x,
                                        )
                #             else:
                #                 reject_counters["invalid_matchups"] += 1
                #         else:
                #             reject_counters["projection_too_low"] += 1
                #     else:
                #         if salary > salary_ceiling:
                #             reject_counters["salary_too_high"] += 1
                #         elif salary < salary_floor:
                #             reject_counters["salary_too_low"] += 1
                # else:
                #     reject_counters["stack_length_insufficient"] += 1
        # return lus, reject_counters
        return lus

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            teams = []
            opponents = []
            matchups = []
            # put def first to make it easier to avoid overlap
            temp_roster_construction = [
                "DST",
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
            ]
            for k in self.player_dict.keys():
                if "Team" not in self.player_dict[k].keys():
                    print(
                        self.player_dict[k]["Name"],
                        " name mismatch between projections and player ids!",
                    )
                ids.append(self.player_dict[k]["ID"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                if self.player_dict[k]["fieldFpts"] >= self.projection_minimum:
                    projections.append(self.player_dict[k]["fieldFpts"])
                else:
                    projections.append(0)
                teams.append(self.player_dict[k]["Team"])
                opponents.append(self.player_dict[k]["Opp"])
                matchups.append(self.player_dict[k]["Matchup"])
                pos_list = []
                for pos in temp_roster_construction:
                    if pos in self.player_dict[k]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                positions.append(np.array(pos_list))
            in_lineup = np.zeros(shape=len(ids))
            ownership = np.array(ownership)
            salaries = np.array(salaries)
            projections = np.array(projections)
            pos_matrix = np.array(positions)
            ids = np.array(ids)
            optimal_score = self.optimal_score
            salary_floor = self.min_lineup_salary
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            stack_usage = self.pct_field_using_stacks
            teams = np.array(teams)
            opponents = np.array(opponents)
            overlap_limit = self.overlap_limit
            problems = []
            stacks = np.random.binomial(n=1, p=self.pct_field_using_stacks, size=diff)
            stack_len = np.random.choice(
                a=[1, 2],
                p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks],
                size=diff,
            )
            max_stack_len = 2
            num_players_in_roster = len(self.roster_construction)
            a = list(self.stacks_dict.keys())
            p = np.array(list(self.stacks_dict.values()))
            probs = p / sum(p)
            stacks = stacks.astype(str)
            for i in range(len(stacks)):
                if stacks[i] == "1":
                    choice = random.choices(a, weights=probs, k=1)
                    stacks[i] = choice[0]
                else:
                    stacks[i] = ""
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                    i,
                    ids,
                    in_lineup,
                    pos_matrix,
                    ownership,
                    salary_floor,
                    salary_ceiling,
                    optimal_score,
                    salaries,
                    projections,
                    max_pct_off_optimal,
                    teams,
                    opponents,
                    stacks[i],
                    stack_len[i],
                    overlap_limit,
                    max_stack_len,
                    matchups,
                    num_players_in_roster,
                    self.site,
                )
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print(
                    "number of running processes =",
                    pool.__dict__["_processes"]
                    if (pool.__dict__["_state"]).upper() == "RUN"
                    else None,
                )
                pool.close()
                pool.join()
            print("pool closed")
            self.update_field_lineups(output,diff)
            end_time = time.time()
            print("lineups took " + str(end_time - start_time) + " seconds")
            print(str(diff) + " field lineups successfully generated")
            # print("Reject counters:", dict(overall_reject_counters))

            # print(self.field_lineups)

    def get_start_time(self, player_id):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                matchup = player["Matchup"]
                return self.game_info[matchup]
        return None

    def get_player_attribute(self, player_id, attribute):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                return player.get(attribute, None)
        return None

    def is_valid_for_position(self, player, position_idx):
        return any(
            pos in self.position_map[position_idx]
            for pos in self.get_player_attribute(player, "Position")
        )


    def sort_lineup_by_start_time(self, lineup):
        flex_index = 8  # Assuming FLEX is at index 8
        flex_player = lineup[flex_index]
        flex_player_start_time = self.get_start_time(flex_player)

        # Initialize variables to track the best swap candidate
        latest_start_time = flex_player_start_time
        swap_candidate_index = None

        # Iterate over RB, WR, and TE positions (indices 2 to 7)
        for i in range(2, 8):
            current_player = lineup[i]
            current_player_start_time = self.get_start_time(current_player)

            # Update the latest start time and swap candidate index
            if (current_player_start_time and current_player_start_time > latest_start_time and
                self.is_valid_for_position(flex_player, i) and
                self.is_valid_for_position(current_player, flex_index)):

                latest_start_time = current_player_start_time
                swap_candidate_index = i

        # Perform the swap if a suitable candidate is found
        if swap_candidate_index is not None:
            #print(f"Swapping: {lineup[swap_candidate_index]} with {flex_player}")
            lineup[flex_index], lineup[swap_candidate_index] = lineup[swap_candidate_index], lineup[flex_index]

        return lineup

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(
                range(
                    max(self.field_lineups.keys()) + 1,
                    max(self.field_lineups.keys()) + 1 + diff,
                )
            )

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_list = sorted(next(iter(o.values()))["Lineup"])
            lineup_set = frozenset(lineup_list)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1

                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1

                # Updating the field lineups dictionary
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    if self.site == "dk":
                        sorted_lineup = self.sort_lineup_by_start_time(
                            next(iter(o.values()))["Lineup"]
                        )
                    else:
                        sorted_lineup = next(iter(o.values()))["Lineup"]


                    self.field_lineups[nk] = next(iter(o.values()))
                    self.field_lineups[nk]["Lineup"] = sorted_lineup
                    self.field_lineups[nk]["Count"] += self.seen_lineups[lineup_set]
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    @staticmethod
    def run_simulation_for_game(
        team1_id,
        team1,
        team2_id,
        team2,
        num_iterations,
        roster_construction,
    ):
        # Define correlations between positions

        def get_corr_value(player1, player2):
            # First, check for specific player-to-player correlations
            if player2["Name"] in player1.get("Player Correlations", {}):
                return player1["Player Correlations"][player2["Name"]]

            # If no specific correlation is found, proceed with the general logic
            position_correlations = {
                "QB": -0.5,
                "RB": -0.2,
                "WR": 0.1,
                "TE": -0.2,
                "K": -0.5,
                "DST": -0.5,
            }

            if player1["Team"] == player2["Team"] and player1["Position"][0] == player2["Position"][0]:
                primary_position = player1["Position"][0]
                return position_correlations[primary_position]

            if player1["Team"] != player2["Team"]:
                player_2_pos = "Opp " + str(player2["Position"][0])
            else:
                player_2_pos = player2["Position"][0]

            return player1["Correlations"].get(
                player_2_pos, 0
            )  # Default to 0 if no correlation is found

        def build_covariance_matrix(players):
            N = len(players)
            matrix = [[0 for _ in range(N)] for _ in range(N)]
            corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = (
                            players[i]["StdDev"] ** 2
                        )  # Variance on the diagonal
                        corr_matrix[i][j] = 1
                    else:
                        matrix[i][j] = (
                            get_corr_value(players[i], players[j])
                            * players[i]["StdDev"]
                            * players[j]["StdDev"]
                        )
                        corr_matrix[i][j] = get_corr_value(players[i], players[j])
            return matrix, corr_matrix

        def ensure_positive_semidefinite(matrix):
            eigs = np.linalg.eigvals(matrix)
            if np.any(eigs < 0):
                jitter = abs(min(eigs)) + 1e-6  # a small value
                matrix += np.eye(len(matrix)) * jitter
            return matrix

        game = team1 + team2
        covariance_matrix, corr_matrix = build_covariance_matrix(game)
        # print(team1_id, team2_id)
        # print(corr_matrix)
        corr_matrix = np.array(corr_matrix)

        # Given eigenvalues and eigenvectors from previous code
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Set negative eigenvalues to zero
        eigenvalues[eigenvalues < 0] = 0

        # Reconstruct the matrix
        covariance_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)

        try:
            samples = multivariate_normal.rvs(
                mean=[player["Fpts"] for player in game],
                cov=covariance_matrix,
                size=num_iterations,
            )
        except:
            print(team1_id, team2_id, "bad matrix")

        player_samples = []
        for i, player in enumerate(game):
            if "QB" in player["Position"]:
                sample = samples[:, i]
            else:
                sample = samples[:, i]
            # if player['Team'] in ['LAR','SEA']:
            #     print(player['Name'], player['Fpts'], player['StdDev'], sample, np.mean(sample), np.std(sample))
            player_samples.append(sample)

        temp_fpts_dict = {}
        # print(team1_id, team2_id, len(game), uniform_samples.T.shape, len(player_samples), covariance_matrix.shape )

        for i, player in enumerate(game):
            temp_fpts_dict[player["ID"]] = player_samples[i]

        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, figsize=(15, 25))
        # fig.tight_layout(pad=5.0)

        # for i, player in enumerate(game):
        #     sns.kdeplot(player_samples[i], ax=ax1, label=player['Name'])

        # ax1.legend(loc='upper right', fontsize=14)
        # ax1.set_xlabel('Fpts', fontsize=14)
        # ax1.set_ylabel('Density', fontsize=14)
        # ax1.set_title(f'Team {team1_id}{team2_id} Distributions', fontsize=14)
        # ax1.tick_params(axis='both', which='both', labelsize=14)

        # y_min, y_max = ax1.get_ylim()
        # ax1.set_ylim(y_min, y_max*1.1)

        # ax1.set_xlim(-5, 50)

        # # # Sorting players and correlating their data
        # player_names = [f"{player['Name']} ({player['Position']})" if player['Position'] is not None else f"{player['Name']} (P)" for player in game]

        # # # Ensuring the data is correctly structured as a 2D array
        # sorted_samples_array = np.array(player_samples)
        # if sorted_samples_array.shape[0] < sorted_samples_array.shape[1]:
        #     sorted_samples_array = sorted_samples_array.T

        # correlation_matrix = pd.DataFrame(np.corrcoef(sorted_samples_array.T), columns=player_names, index=player_names)

        # sns.heatmap(correlation_matrix, annot=True, ax=ax2, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax2.set_title(f'Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # original_corr_matrix = pd.DataFrame(corr_matrix, columns=player_names, index=player_names)
        # sns.heatmap(original_corr_matrix, annot=True, ax=ax3, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax3.set_title(f'Original Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # cov_matrix = pd.DataFrame(covariance_matrix, columns=player_names, index=player_names)
        # sns.heatmap(cov_matrix, annot=True, ax=ax4, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax4.set_title(f'Original Covariance Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # plt.savefig(f'output/Team_{team1_id}{team2_id}_Distributions_Correlation.png', bbox_inches='tight')
        # plt.close()

        return temp_fpts_dict
    
    @staticmethod
    @jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array    

    def run_tournament_simulation(self):
        print("Running " + str(self.num_iterations) + " simulations")
        for f in self.field_lineups:
            if len(self.field_lineups[f]["Lineup"]) != 9:
                print("bad lineup", f, self.field_lineups[f])
        print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        start_time = time.time()
        temp_fpts_dict = {}
        size = self.num_iterations
        game_simulation_params = []
        for m in self.matchups:
            game_simulation_params.append(
                (
                    m[0],
                    self.teams_dict[m[0]],
                    m[1],
                    self.teams_dict[m[1]],
                    self.num_iterations,
                    self.roster_construction,
                )
            )
        with mp.Pool() as pool:
            results = pool.starmap(self.run_simulation_for_game, game_simulation_params)

        for res in results:
            temp_fpts_dict.update(res)

        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + "seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            # if index == 0:
            #    print(x)
            lu_type = x["Type"]
            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            qb_stack = 0
            qb_tm = ""
            players_vs_def = 0
            def_opps = []
            simDupes = x['Count']
            for id in x["Lineup"]:
                for k, v in self.player_dict.items():
                    if v["ID"] == id:
                        if "DST" in v["Position"]:
                            def_opps.append(v["Opp"])
                        if "QB" in v["Position"]:
                            qb_tm = v["Team"]
            for id in x["Lineup"]:
                for k, v in self.player_dict.items():
                    if v["ID"] == id:
                        salary += v["Salary"]
                        fpts_p += v["Fpts"]
                        fieldFpts_p += v["fieldFpts"]
                        ceil_p += v["Ceiling"]
                        own_p.append(v["Ownership"] / 100)
                        lu_names.append(v["Name"])
                        if "DST" not in v["Position"]:
                            lu_teams.append(v["Team"])
                            if v["Team"] in def_opps:
                                players_vs_def += 1
                        continue
            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            # Find the QB team in stacks and set it as primary stack, remove it from stacks and subtract 1 to make sure qb isn't counted
            for s in stacks:
                if s[0] == qb_tm:
                    primaryStack = str(qb_tm) + " " + str((s[1]))
                    stacks.remove(s)
                    break

            # After removing QB team, the first team in stacks will be the team with most players not in QB stack
            secondaryStack = str(stacks[0][0]) + " " + str(stacks[0][1])
            own_p = np.prod(own_p)
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / x['Count'] / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},${},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        roi_p,
                        own_p,
                        roi_round,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes,
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{}%,{}%,{}%,{},{},{},{},{}".format(
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / x['Count'] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / x['Count'] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        x["Lineup"][1],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][0],
                        lu_names[0].replace("#", "-"),
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        roi_p,
                        own_p,
                        roi_round,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{}%,{}%,{},{},{},{},{},{}".format(
                        x["Lineup"][1],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][0],
                        lu_names[0].replace("#", "-"),
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%, Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Position,Team,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val["Lineup"]:
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Top1Percent": val["Top1Percent"],
                            "In": val['Count'],
                            "ROI": val["ROI"],
                        }
                    else:
                        unique_players[player]["Wins"] = (
                            unique_players[player]["Wins"] + val["Wins"]
                        )
                        unique_players[player]["Top1Percent"] = (
                            unique_players[player]["Top1Percent"] + val["Top1Percent"]
                        )
                        unique_players[player]["In"] = unique_players[player]["In"] + val['Count']
                        unique_players[player]["ROI"] = (
                            unique_players[player]["ROI"] + val["ROI"]
                        )
            top1PercentCount = (0.01) * self.field_size
            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top1Percent"] / top1PercentCount / self.num_iterations  * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if player == v["ID"]:
                        proj_own = v["Ownership"]
                        p_name = v["Name"]
                        position = "/".join(v.get("Position"))
                        team = v.get("Team")
                        break
                f.write(
                    "{},{},{},{}%,{}%,{}%,{}%,${}\n".format(
                        p_name.replace("#", "-"),
                        position,
                        team,
                        win_p,
                        top10_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )
