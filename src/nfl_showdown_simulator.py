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
#import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
import sys

class NFL_Showdown_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    seen_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
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

        #ownership_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["ownership_path"]),
        #)
        #self.load_ownership(ownership_path)

        #boom_bust_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        #)
        #self.load_boom_bust(boom_bust_path)
                
 #       batting_order_path = os.path.join(
 #           os.path.dirname(__file__),
#            "../{}_data/{}".format(site, self.config["batting_order_path"]),
#        )                
#        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.salary = 50000
            self.roster_construction = ['CPT', 'FLEX', 'FLEX', 'FLEX', 'FLEX', 'FLEX']
        
        elif site == "fd":
            self.salary = 60000
            self.roster_construction = ['CPT', 'FLEX', 'FLEX', 'FLEX', 'FLEX']

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
        
        #self.adjust_default_stdev()
        self.assertPlayerDict()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        #if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
        #self.generate_field_lineups()
        self.load_correlation_rules()
    
    #make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config['pct_field_using_stacks'])
        self.default_qb_var = float(self.config['default_qb_var'])
        self.default_skillpos_var = float(self.config['default_skillpos_var'])
        self.default_def_var = float(self.config['default_def_var'])
        self.overlap_limit = float(self.config['num_players_vs_def'])
        self.pct_field_double_stacks = float(self.config['pct_field_double_stacks'])
        self.correlation_rules = self.config["custom_correlations"]
        self.allow_def_vs_qb_cpt = self.config["allow_def_vs_qb_cpt"]
    
    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s['ID'] == '' or s['ID'] is None:
                print(s['Name'] + ' name mismatch between projections and player ids, excluding from player_dict')
                self.player_dict.pop(p)
        
    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
            #print(s['Name'],s['ID'])
        #print(self.player_dict)
        problem = plp.LpProblem('NFL', plp.LpMaximize)
        lp_variables = {self.player_dict[(player, pos_str, team)]['ID']: plp.LpVariable(
            str(self.player_dict[(player, pos_str, team)]['ID']), cat='Binary') for (player, pos_str, team) in self.player_dict}
        
        # set the objective - maximize fpts
        problem += plp.lpSum(self.player_dict[(player, pos_str, team)]['fieldFpts'] * lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                             for (player, pos_str, team) in self.player_dict), 'Objective'

        # Set the salary constraints
        problem += plp.lpSum(self.player_dict[(player, pos_str, team)]['Salary'] * lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                             for (player, pos_str, team) in self.player_dict) <= self.salary
        
        player_id_dict = {(player, team): player_data['ID'] for (player, pos_str, team), player_data in self.player_dict.items()}
    
        for (player, pos_str, team) in self.player_dict:
            if pos_str == 'CPT':
                # Get the player's ID in both the CPT and FLEX positions
                cpt_id = self.player_dict[(player, 'CPT', team)]['ID']
                flex_id = player_id_dict.get((player, team))
                
                if flex_id:
                    # Add a constraint to ensure that the player can't be selected as both CPT and FLEX
                    problem += lp_variables[cpt_id] + lp_variables[flex_id] <= 1


        if self.site == 'dk':
            # Can have exactly one CPT
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict  if 'CPT' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 4 UTIL
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict  if 'FLEX' in self.player_dict[(player, pos_str, team)]['Position']) == 5
            # Can only roster 5 total players
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict ) == 6
            # Max 4 per team
            for team in self.team_list:
                problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                     for (player, pos_str, team) in self.player_dict if self.player_dict[(player, pos_str, team)]['Team'] == team) <= 5
        
        elif self.site == 'fd':
            # Can have exactly one CPT
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict  if 'CPT' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 4 UTIL
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict  if 'FLEX' in self.player_dict[(player, pos_str, team)]['Position']) == 4
            # Can only roster 5 total players
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                 for (player, pos_str, team) in self.player_dict ) == 5
            # Max 4 per team
            for team in self.team_list:
                problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                     for (player, pos_str, team) in self.player_dict if self.player_dict[(player, pos_str, team)]['Team'] == team) <= 4

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
            print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                len(self.num_lineups), self.num_lineups))
        except TypeError:
            for p,s in self.player_dict.items():
                if s["ID"]==0:
                    print(s["Name"] + ' name mismatch between projections and player ids')
                if s['ID'] == '':
                    print(s['Name'] + ' name mismatch between projections and player ids')
                if s['ID'] is None:
                    print(s['Name'])
        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)
        
    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                player_name = row[name_key].replace("-", "#").lower().strip()
                #if qb and dst not in position add flex
                team_key = "teamabbrev" if self.site == "dk" else "team"
                team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                match = re.search(pattern='(\w{2,4}@\w{2,4})', string=row[game_info])
                if match:
                    opp = match.groups()[0].split('@')
                    self.matchups.add((opp[0], opp[1]))
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                #if not opp:
                #    print(row)
                if self.site == "dk":
                    position_key = "roster position" 
                    position = row[position_key]
                    pos_str = str(position)
                    if (player_name,pos_str, team) in self.player_dict:
                        self.player_dict[(player_name,pos_str, team)]["ID"] = str(row["id"])
                        self.player_dict[(player_name,pos_str, team)]["Team"] =  row[team_key]
                        self.player_dict[(player_name,pos_str, team)]["Opp"] = team_opp
                        self.player_dict[(player_name,pos_str, team)]["Matchup"] = opp
                elif self.site == "fd":
                    for position in ['CPT','FLEX']:
                        if (player_name,position, team) in self.player_dict:
                            self.player_dict[(player_name,pos_str, team)]["ID"] = str(row["id"])
                            self.player_dict[(player_name,pos_str, team)]["Team"] =  row[team_key]
                            self.player_dict[(player_name,pos_str, team)]["Opp"] = team_opp
                            self.player_dict[(player_name,pos_str, team)]["Matchup"] = opp
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
        if len(self.correlation_rules.keys())>0:
            for c in self.correlation_rules.keys():
                for k in self.player_dict:
                    if c.replace("-", "#").lower().strip() in self.player_dict[k].values():
                        for v in self.correlation_rules[c].keys():
                            self.player_dict[k]['Correlations'][v] = self.correlation_rules[c][v]
                            
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
                    print('unable to load player fpts: ' + player_name + ', fpts:' + row['fpts'])
                if 'fieldfpts' in row:
                    if row['fieldfpts'] == '':
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row['fieldfpts'])
                else:
                    fieldFpts = fpts
                if fpts == 0:
                    continue
                position = [pos for pos in row['position'].split('/')]
                position.sort()
                #if qb and dst not in position add flex
                if self.site == 'fd':
                    if 'D' in position:
                        position = ['DST']
                pos = position[0]
                if 'stddev' in row:
                    if row['stddev'] == '' or float(row['stddev']) == 0:
                        if position == 'QB':
                            stddev = fpts* self.default_qb_var
                        elif position == 'DST':
                            stddev = fpts* self.default_def_var
                        else:
                            stddev = fpts*self.default_skillpos_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    if position == 'QB':
                        stddev = fpts* self.default_qb_var
                    elif position == 'DST':
                        stddev = fpts* self.default_def_var
                    else:
                        stddev = fpts*self.default_skillpos_var
                #check if ceiling exists in row columns
                if 'ceiling' in row:
                    if row['ceiling'] == '' or float(row['ceiling'])==0:
                        ceil = fpts+stddev
                    else:
                        ceil = float(row['ceiling'])
                else:
                    ceil = fpts+stddev
                if row['salary']:
                    sal = int(row['salary'].replace(",", ""))
                if pos == 'QB':
                    corr = {'QB': 1, 'RB': 0.08, 'WR': 0.62, 'TE': 0.32, 'K': -0.02, 'DST' : -0.09, 'Opp QB': 0.24, 'Opp RB' : 0.04, 'Opp WR': 0.19, 'Opp TE' : 0.1, 'Opp K': -0.03, 'Opp DST': -0.41}
                elif pos == 'RB':
                    corr = {'QB': 0.08, 'RB': 1, 'WR': -0.09, 'TE': -0.02, 'K': 0.16, 'DST' : 0.07, 'Opp QB': 0.04, 'Opp RB' : -0.08, 'Opp WR': 0.01, 'Opp TE' : 0.03, 'Opp K': -0.13, 'Opp DST': -0.33}
                elif pos == 'WR':
                    corr = {'QB': 0.62, 'RB': -0.09, 'WR': 1, 'TE': -0.07, 'K':0.0, 'DST' : -0.08, 'Opp QB': 0.19, 'Opp RB' : 0.01, 'Opp WR': 0.16, 'Opp TE' : 0.08, 'Opp K': 0.06, 'Opp DST': -0.22}
                elif pos == 'TE':
                    corr = {'QB': 0.32, 'RB': -0.02, 'WR': -0.07, 'TE': 1, 'K': 0.02, 'DST' : -0.08, 'Opp QB': 0.1, 'Opp RB' : 0.03, 'Opp WR': 0.08, 'Opp TE' : 0, 'Opp K': 0, 'Opp DST': -0.14}
                elif pos == 'K':
                    corr = {'QB': -0.02, 'RB':0.16, 'WR':0, 'TE':0.02, 'K':1, 'DST': 0.23, 'Opp QB': -0.03, 'Opp RB' : -0.13, 'Opp WR': 0.06, 'Opp TE' : 0, 'Opp K': -0.04, 'Opp DST': -0.32}
                elif pos == 'DST':
                    corr = {'QB': -0.09, 'RB': 0.07, 'WR': -0.08, 'TE': -0.08, 'K': 0.23, 'DST' : 1, 'Opp QB': -0.41, 'Opp RB' : -0.33, 'Opp WR': -0.22, 'Opp TE' : -0.14, 'Opp K': -0.32, 'Opp DST': -0.27}
                team = row['team']
                if team == 'LA':
                    team = 'LAR'
                if self.site == 'fd':
                    if team == 'JAX':
                        team = 'JAC'
                own = float(row['own%'].replace('%','')) 
                if own == 0:
                    own = 0.1
                if 'cptown%' in row:
                    cptOwn = float(row['cptown%'].replace('%','')) 
                    if cptOwn == 0:
                        cptOwn = 0.1
                else:
                    cptOwn = own*.5
                sal = int(row["salary"].replace(",", ""))
                pos_str = 'FLEX'
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts" : fieldFpts,
                    "Position": position,
                    'rosterPosition': 'FLEX',
                    "Name" : player_name,
                    "Team" : team,
                    "Opp" : '',
                    "ID": '',
                    "Salary": sal,
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations" : corr,
                    "In Lineup": False
                }
                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[(player_name, pos_str, team)].get("Opp", '')
                    player_data["ID"] = self.player_dict[(player_name, pos_str, team)].get("ID", '')

                self.player_dict[(player_name, pos_str,team)] = player_data
                self.teams_dict[team].append(player_data)
                pos_str = 'CPT'
                player_data = {
                    "Fpts": 1.5*fpts,
                    "fieldFpts" : 1.5*fieldFpts,
                    "Position": position,
                    'rosterPosition': 'CPT',
                    "Name" : player_name,
                    "Team" : team,
                    "Opp" : '',
                    "ID": '',
                    "Salary": 1.5*sal,
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": cptOwn,
                    "Correlations" : corr,
                    "In Lineup": False
                }                
                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[(player_name, pos_str, team)].get("Opp", '')
                    player_data["ID"] = self.player_dict[(player_name, pos_str, team)].get("ID", '')

                self.player_dict[(player_name, pos_str,team)] = player_data
                #self.teams_dict[team].append(player_data)  # Add player data to their respective team

    def load_team_stacks(self):
        # Initialize a dictionary to hold QB ownership by team
        qb_ownership_by_team = {}
        
        for p in self.player_dict:
            # Check if player is a QB
            if 'QB' in self.player_dict[p]['Position']:
                # Fetch the team of the QB
                team = self.player_dict[p]['Team']
                
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

    def extract_id(self,cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
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
                lineup = [self.extract_id(str(row[j])) for j in range(len(self.roster_construction))]                      
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]['ID'] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i,l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]['ID'] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i,l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                if not error:
                    self.field_lineups[j] = { "Lineup": {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top10": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "input",
                    }, "count":1}
                    j += 1
        print("loaded {} lineups".format(j))
        #print(self.field_lineups)

    @staticmethod
    def select_player(pos, in_lineup, ownership, ids, player_dict, def_opp=None, teams=None):
        valid_players = np.where((pos > 0) & (in_lineup == 0) & ((teams != def_opp) if def_opp is not None else True))
        plyr_list = ids[valid_players]
        prob_list = ownership[valid_players] / ownership[valid_players].sum()
        choice = np.random.choice(a=plyr_list, p=prob_list)
        return np.where(ids == choice)[0], choice

    @staticmethod
    def validate_lineup(salary, salary_floor, salary_ceiling, proj, optimal_score, max_pct_off_optimal, player_teams):
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)

        if salary_floor <= salary <= salary_ceiling and proj >= reasonable_projection and len(set(player_teams)) > 1:
            return True
        return False

    @staticmethod
    def generate_lineups(lu_num, ids, in_lineup, pos_matrix, ownership, salary_floor, salary_ceiling, optimal_score, salaries, projections, max_pct_off_optimal, teams, opponents, overlap_limit, matchups, new_player_dict):
        np.random.seed(lu_num)
        lus = {}
        in_lineup.fill(0)
        iteration_count = 0
        while True:
            iteration_count += 1
            salary, proj = 0, 0
            lineup, player_teams, lineup_matchups = [], [], []
            def_opp, players_opposing_def, cpt_selected = None, 0, False
            in_lineup.fill(0)
            cpt_name = None

            for k, pos in enumerate(pos_matrix.T):
                position_constraint = k >= 1 and players_opposing_def < overlap_limit
                choice_idx, choice = NFL_Showdown_Simulator.select_player(pos, in_lineup, ownership, ids, def_opp if position_constraint else None, teams if position_constraint else None)
                
                if k == 0:
                    cpt_player_info = new_player_dict[choice]
                    flex_choice_idx = next((i for i, v in enumerate(new_player_dict.values()) if v["Name"] == cpt_player_info["Name"] and v["Team"] == cpt_player_info["Team"] and v["Position"] == cpt_player_info["Position"] and v["rosterPosition"] == "FLEX"), None)
                    if flex_choice_idx is not None:
                        in_lineup[flex_choice_idx] = 1
                    def_opp = opponents[choice_idx][0]
                    cpt_selected = True
                
                if cpt_selected and "QB" in new_player_dict[choice]["Position"] and "DEF" in [new_player_dict[x]["Position"] for x in lineup]:
                    continue
                
                lineup.append(str(choice))
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]



                #lineup_matchups.append(matchups[choice_idx[0]])
                player_teams.append(teams[choice_idx][0])
                
                if teams[choice_idx][0] == def_opp:
                    players_opposing_def += 1

            if NFL_Showdown_Simulator.validate_lineup(salary, salary_floor, salary_ceiling, proj, optimal_score, max_pct_off_optimal, player_teams):
                lus[lu_num] = {
                    "Lineup": lineup,
                    "Wins": 0,
                    "Top10": 0,
                    "ROI": 0,
                    "Cashes": 0,
                    "Type": "generated_nostack",
                }
                break
        return lus

    
    def remap_player_dict(self, player_dict):
        remapped_dict = {}
        for key, value in player_dict.items():
            if 'ID' in value:
                player_id = value['ID']
                remapped_dict[player_id] = value
            else:
                raise KeyError(f"Player details for {key} does not contain an 'ID' key")
        return remapped_dict

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(f"supplied lineups >= contest field size. only retrieving the first {self.field_size} lineups")
            return

        print(f'Generating {diff} lineups.')
        player_data = self.extract_player_data()
        
        # Initialize problem list
        problems = self.initialize_problems_list(diff, player_data)
        
        # Handle stacks logic
        #stacks = self.handle_stacks_logic(diff)
        #print(problems)
        
        start_time = time.time()
        
        # Parallel processing for generating lineups
        with mp.Pool() as pool:
            output = pool.starmap(self.generate_lineups, problems)
            pool.close()
            pool.join()
        
        print('pool closed')
        
        # Update field lineups
        self.update_field_lineups(output, diff)
        
        end_time = time.time()
        print(f"lineups took {end_time - start_time} seconds")
        print(f"{diff} field lineups successfully generated")


    def extract_player_data(self):
        ids, ownership, salaries, projections, teams, opponents, matchups, positions = [], [], [], [], [], [], [], []
        for player_info in self.player_dict.values():
            if 'Team' not in player_info:
                print(f"{player_info['Name']} name mismatch between projections and player ids!")
            ids.append(player_info['ID'])
            ownership.append(player_info['Ownership'])
            salaries.append(player_info['Salary'])
            projections.append(max(0, player_info.get('fieldFpts', 0)))
            teams.append(player_info['Team'])
            opponents.append(player_info['Opp'])
            matchups.append(player_info['Matchup'])
            pos_list = [1 if pos in player_info['rosterPosition'] else 0 for pos in self.roster_construction]
            positions.append(np.array(pos_list))
        return ids, ownership, salaries, projections, teams, opponents, matchups, positions


    def initialize_problems_list(self, diff, player_data):
        ids, ownership, salaries, projections, teams, opponents, matchups, positions = player_data
        in_lineup = np.zeros(shape=len(ids))
        ownership, salaries, projections, pos_matrix = map(np.array, [ownership, salaries, projections, positions])
        teams, opponents, ids = map(np.array, [teams, opponents, ids])
        new_player_dict = self.remap_player_dict(self.player_dict)
        problems = []
        for i in range(diff):
            lu_tuple = (i, ids, in_lineup, pos_matrix, ownership, self.min_lineup_salary, self.salary, self.optimal_score, salaries, projections, self.max_pct_off_optimal, teams, opponents, self.overlap_limit, matchups, new_player_dict)
            problems.append(lu_tuple)
        #print(self.player_dict.keys())
        return problems

    def handle_stacks_logic(self, diff):
        stacks = np.random.binomial(n=1, p=self.pct_field_using_stacks, size=diff).astype(str)
        stack_len = np.random.choice(a=[1, 2], p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks], size=diff)
        a = list(self.stacks_dict.keys())
        p = np.array(list(self.stacks_dict.values()))
        probs = p / sum(p)
        for i in range(len(stacks)):
            if stacks[i] == '1':
                choice = random.choices(a, weights=probs, k=1)
                stacks[i] = choice[0]
            else:
                stacks[i] = ''
        return stacks

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(range(max(self.field_lineups.keys()) + 1, max(self.field_lineups.keys()) + 1 + diff))

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_tuple = tuple(next(iter(o.values()))['Lineup'])   # Getting the list associated with the 'Lineup' key and converting it to a tuple
            
            # Keeping track of lineup duplication counts
            if lineup_tuple in self.seen_lineups:
                self.seen_lineups[lineup_tuple] += 1
            else:
                self.seen_lineups[lineup_tuple] = 1

            # Updating the field lineups dictionary
            if nk in self.field_lineups.keys():
                print("bad lineups dict, please check dk_data files")
            else:
                self.field_lineups[nk] = {"Lineup": next(iter(o.values())), "count": self.seen_lineups[lineup_tuple]}
                nk += 1


    def calc_gamma(self,mean,sd):
        alpha = (mean / sd) ** 2
        beta = sd ** 2 / mean
        return alpha,beta

    def run_simulation_for_game(self, team1_id, team1, team2_id, team2, num_iterations):
        def get_corr_value(player1, player2):
            if player1['Team'] == player2['Team'] and player1['Position'][0] == player2['Position'][0]:
                return -0.25
            if player1['Team'] != player2['Team']:
                player_2_pos = 'Opp ' + str(player2['Position'][0])
            else:
                player_2_pos = player2['Position'][0]
            return player1['Correlations'].get(player_2_pos, 0)

        def build_covariance_matrix(players):
            N = len(players)
            matrix = [[0 for _ in range(N)] for _ in range(N)]
            corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = players[i]['StdDev'] ** 2  # Variance on the diagonal
                        corr_matrix[i][j] = 1
                    else:
                        matrix[i][j] = get_corr_value(players[i], players[j]) * players[i]['StdDev'] * players[j]['StdDev']
                        corr_matrix[i][j] = get_corr_value(players[i], players[j])
            return matrix, corr_matrix

        def ensure_positive_semidefinite(matrix):
            eigs = np.linalg.eigvals(matrix)
            if np.any(eigs < 0):
                jitter = abs(min(eigs)) + 1e-6  # a small value
                matrix += np.eye(len(matrix)) * jitter

            # Given eigenvalues and eigenvectors from previous code
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)

            # Set negative eigenvalues to zero
            eigenvalues[eigenvalues < 0] = 0

            # Reconstruct the matrix
            matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            return matrix

        # Filter out players with projections less than or equal to 0
        team1 = [player for player in team1 if player['Fpts'] > 0 and player['rosterPosition'] == 'FLEX']
        team2 = [player for player in team2 if player['Fpts'] > 0 and player['rosterPosition'] == 'FLEX']

        game = team1 + team2
        covariance_matrix, corr_matrix = build_covariance_matrix(game)
        covariance_matrix = ensure_positive_semidefinite(covariance_matrix)

        try:
            samples = multivariate_normal.rvs(mean=[player['Fpts'] for player in game], cov=covariance_matrix, size=num_iterations)
        except Exception as e:
            print(f"{team1_id}, {team2_id}, bad matrix: {str(e)}")
            return {}

        player_samples = [samples[:, i] for i in range(len(game))]
        
        temp_fpts_dict = {}
        for i, player in enumerate(game):
            temp_fpts_dict[player['ID']] = player_samples[i]

        return temp_fpts_dict
    
    @staticmethod
    def process_simulation_chunk(args):
        
        simulation_indices, ranks, fpts_array, payout_array, entry_fee, field_lineups_keys, use_contest_data = args
        #print('starting chunk:' + str(simulation_indices))
        result_dict = {}
        num_simulations = len(simulation_indices)
        for simulation_index in range(num_simulations):
            ranks_in_sim = ranks[:, simulation_index]
            scores_in_sim = fpts_array[:, simulation_index]

            unique, counts = np.unique(scores_in_sim, return_counts=True)
            score_to_count = dict(zip(unique, counts))

            payouts_in_sim = np.full(len(field_lineups_keys), -entry_fee)
            
        for idx, score in enumerate(unique):
            tie_count = score_to_count[score]
            indices = np.where(scores_in_sim == score)[0]

            if tie_count == 1:
                payouts_in_sim[indices] = payout_array[indices]
            else:
                total_payout = sum(payout_array[i] for i in indices)
                average_payout = total_payout / tie_count
                payouts_in_sim[indices] = average_payout

            # Using numpy array instead of iterating through keys can help speed up the code
        if use_contest_data:
            results_in_sim = payouts_in_sim[ranks_in_sim]
            for idx, roi in zip(field_lineups_keys, results_in_sim):
                if idx not in result_dict:
                    result_dict[idx] = 0
                result_dict[idx] += roi

        #print('chunk done:' + str(simulation_indices))
        return result_dict

    def run_tournament_simulation(self):
        print(f"Running {self.num_iterations} simulations")
        
        def generate_cpt_outcomes(flex_dict):
            cpt_dict = {}
            for player_id, flex_outcomes in flex_dict.items():
                # Fetch team information using the player_id
                # Assuming self.player_dict uses a structure like {(player_name, position, team): player_data}
                player_data_flex = [data for (name, pos, team), data in self.player_dict.items() if data['ID'] == player_id and pos == 'FLEX']
                if player_data_flex:
                    player_data_flex = player_data_flex[0]  # Get the first match (there should only be one)
                    team = player_data_flex['Team']
                    
                    # Fetch the CPT data using the player_name and team fetched from the above step
                    player_data_cpt = self.player_dict.get((player_data_flex['Name'], 'CPT', team))
                    if player_data_cpt:
                        cpt_outcomes = flex_outcomes * 1.5
                        cpt_dict[player_data_cpt['ID']] = cpt_outcomes
            return cpt_dict

        # Validation on lineups
        for f in self.field_lineups:
            if len(self.field_lineups[f]['Lineup']['Lineup']) != len(self.roster_construction):
                print('bad lineup', f, self.field_lineups[f])

        start_time = time.time()
        temp_fpts_dict = {}

        # Get the only matchup since it's a showdown
        matchup = list(self.matchups)[0]  

        # Prepare the arguments for the simulation function
        game_simulation_params = (matchup[0], self.teams_dict[matchup[0]], matchup[1], self.teams_dict[matchup[1]], self.num_iterations)

        # Run the simulation for the single game
        temp_fpts_dict.update(self.run_simulation_for_game(*game_simulation_params))
        cpt_outcomes_dict = generate_cpt_outcomes(temp_fpts_dict)
        temp_fpts_dict.update(cpt_outcomes_dict)
      # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(self.field_size, self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        #print(self.field_lineups)
        #print(temp_fpts_dict)
        #print(payout_array)
        #print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]["Lineup"]])
            except KeyError:
                for player in values["Lineup"]["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        #for k,v in self.player_dict.items():
                            #if v['ID'] == player:
                        #        print(k,v)
                #print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim
        
        fpts_array = fpts_array.astype(np.float16)
        #ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint16)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        t10, t10_counts = np.unique(ranks[0:9], return_counts=True)
        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(shape=self.field_size - len(payout_array), fill_value=-self.entry_fee)
        payout_array = np.concatenate((payout_array, l_array))        
        # Adjusted ROI calculation

        # Split the simulation indices into chunks
        chunk_size = self.num_iterations // 3 # Adjust chunk size as needed
        simulation_chunks = [
            (range(i, min(i+chunk_size, self.num_iterations)), ranks[:, i:min(i+chunk_size, self.num_iterations)].copy(), fpts_array[:, i:min(i+chunk_size, self.num_iterations)].copy(), payout_array, self.entry_fee, list(self.field_lineups.keys()), self.use_contest_data) 
            for i in range(0, self.num_iterations, chunk_size)
        ]
        #print(chunk_size, len(simulation_chunks), simulation_chunks[0])
        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.process_simulation_chunk, simulation_chunks)
        
        # Combine results
        combined_result_dict = {}
        for result_dict in results:
            for idx, roi in result_dict.items():
                if idx not in combined_result_dict:
                    combined_result_dict[idx] = 0
                combined_result_dict[idx] += roi

        # Update field_lineups with combined results
        for idx, roi in combined_result_dict.items():
            self.field_lineups[idx]["Lineup"]["ROI"] += roi
        # Update Wins and Top10
        
        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Lineup"]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in t10:
                self.field_lineups[idx]["Lineup"]["Top10"] += t10_counts[np.where(t10 == idx)][0]
        
        #print(self.field_lineups)
        end_time = time.time()
        diff = end_time - start_time
        print(str(self.num_iterations) + " tournament simulations finished in " + str(diff) + " seconds. Outputting.")

    def output(self):
        unique = {}
        for index, data in self.field_lineups.items():
            lineup = data["Lineup"]["Lineup"]
            lineup_data = data["Lineup"]
            lu_type = lineup_data["Type"]

            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            cpt_tm = ''
            def_opps = []
            players_vs_def = 0

            player_dict_values = {v["ID"]: v for k, v in self.player_dict.items()}

            for player_id in lineup:
                player_data = player_dict_values.get(player_id, {})
                if player_data:
                    if 'DST' in player_data["Position"]:
                        def_opps.append(player_data['Opp'])
                    if 'CPT' in player_data['rosterPosition']:
                        cpt_tm = player_data['Team']

                    salary += player_data.get("Salary", 0)
                    fpts_p += player_data.get("Fpts", 0)
                    fieldFpts_p += player_data.get("fieldFpts", 0)
                    ceil_p += player_data.get("Ceiling", 0)
                    own_p.append(player_data.get("Ownership", 0) / 100)
                    lu_names.append(f"{player_data.get('Name', '')} ({player_data.get('ID', '')})")
                    if 'DST' not in player_data["Position"]:
                        lu_teams.append(player_data['Team'])
                        if player_data['Team'] in def_opps:
                            players_vs_def += 1

            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            primary_stack = secondary_stack = ''
            for s in stacks:
                if s[0] == cpt_tm:
                    primary_stack = f"{cpt_tm} {s[1]}"
                    stacks.remove(s)
                    break

            if stacks:
                secondary_stack = f"{stacks[0][0]} {stacks[0][1]}"

            own_p = np.prod(own_p)
            win_p = round(lineup_data["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(lineup_data["Top10"] / self.num_iterations * 100, 2)
            cash_p = round(lineup_data["Cashes"] / self.num_iterations * 100, 2)
            num_dupes = data['count']
            if self.use_contest_data:
                roi_p = round(
                    lineup_data["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                )
                roi_round = round(lineup_data["ROI"] / self.num_iterations, 2)

            if self.use_contest_data:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{players_vs_def},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{roi_p}%,${roi_round},{num_dupes}"
            else:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{players_vs_def},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{num_dupes}"              
            unique[lineup_str] = fpts_p  # Changed data["Fpts"] to fpts_p, which contains the accumulated Fpts

        return unique


    def player_output(self):
        #out_path = os.path.join(self.output_dir, f"{self.slate_id}_{self.sport}_{self.site}_player_output.csv")
        # First output file
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write("Player,Roster Position,Position,Team,Win%,Top10%,Sim. Own%,Proj. Own%,Avg. Return\n")
            unique_players = {}
            
            for val in self.field_lineups.values():
                lineup_data = val['Lineup']
                for player_id in lineup_data['Lineup']:
                    if player_id not in unique_players:
                        unique_players[player_id] = {
                            "Wins": lineup_data["Wins"],
                            "Top10": lineup_data["Top10"],
                            "In": 1,
                            "ROI": lineup_data["ROI"],
                        }
                    else:
                        unique_players[player_id]["Wins"] += lineup_data["Wins"]
                        unique_players[player_id]["Top10"] += lineup_data["Top10"]
                        unique_players[player_id]["In"] += 1
                        unique_players[player_id]["ROI"] += lineup_data["ROI"]

            for player_id, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top10"] / self.num_iterations / 10 * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if v["ID"] == player_id:
                        player_info = v
                        break
                proj_own = player_info.get("Ownership", "N/A")
                p_name = player_info.get("Name", "N/A").replace("#", "-")
                sd_position = player_info.get("rosterPosition", ["N/A"])
                position = player_info.get("Position", ["N/A"])[0]
                team = player_info.get("Team", "N/A")

                f.write(f"{p_name},{sd_position},{position},{team},{win_p}%,{top10_p}%,{field_p}%,{proj_own}%,${roi_p}\n")

    def save_results(self):
        unique = self.output()

        # First output file
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        if self.site == 'dk':
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,ROI%,ROI$,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
            else:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")                
        else:
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,ROI,ROI/Entry Fee,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")   
            else:         
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")           
        self.player_output()