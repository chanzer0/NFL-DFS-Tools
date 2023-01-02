import json
import csv
import os
import re
import pulp as plp
import numpy as np


class NFL_Showdown_Optimizer:
    problem = None
    config = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    player_dict = {}
    max_salary = None
    lineups = {}
    team_list = []
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = 0
    randomness_amount = 0
    home_team = None
    away_team = None

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.problem = plp.LpProblem('NFL', plp.LpMaximize)

        self.load_config()
        self.load_rules()

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # print(row)
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                player_name = player_name[:-
                                          1] if row['Position'] == 'DST' else player_name
                player_name = f'_{player_name}_CPT' if row['Roster Position'] == 'CPT' else f'_{player_name}'
                if player_name in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[player_name]['RealID'] = int(
                            row['ID'])
                        self.player_dict[player_name]['ID'] = int(
                            row['ID'][-3:])
                        self.player_dict[player_name]['Matchup'] = row['Game Info'].split(' ')[
                            0]
                        if self.home_team is None or self.away_team is None:
                            self.home_team = row['Game Info'].split(' ')[
                                0].split('@')[0]
                            self.away_team = row['Game Info'].split(' ')[
                                0].split('@')[1]
                    else:
                        self.player_dict[player_name]['RealID'] = int(
                            row['ID'])
                        self.player_dict[player_name]['ID'] = int(
                            row['Id'].split('-')[1])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                player_name = f'_{player_name}'
                if float(row['Projection']) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {'Fpts': 0.1, 'Position': None, 'ID': 0, 'Salary': 100,
                                                 'Name': '', 'RealID': 0, 'StdDev': 0.1, 'Team': '', 'Ownership': 0.1, 'Ceiling': 0.1}
                self.player_dict[player_name]['Fpts'] = float(
                    row['Projection'])
                self.player_dict[player_name]['Salary'] = int(
                    row['Salary'].replace(',', ''))
                self.player_dict[player_name]['Ceiling'] = float(
                    row['Ceiling'])
                self.player_dict[player_name]['StdDev'] = float(
                    row['Ceiling']) - float(row['Projection'])
                self.player_dict[player_name]['Name'] = row['Name']
                self.player_dict[player_name]['Team'] = row['Team']
                self.player_dict[player_name]['Ownership'] = float(
                    row['Total Own']) - float(row['CPT Own'])
                self.player_dict[player_name]['Position'] = 'FLEX'
                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])
                if self.player_dict[player_name]['Ownership'] == 0:
                    self.player_dict[player_name]['Ownership'] = 1

                # CPT
                player_name = f'{player_name}_CPT'
                self.player_dict[player_name] = {'Fpts': 0.1, 'Position': None, 'ID': 0, 'Salary': 100,
                                                 'Name': '', 'RealID': 0, 'StdDev': 0.1, 'Team': '', 'Ownership': 0.1, 'Ceiling': 0.1}
                self.player_dict[player_name]['Fpts'] = float(
                    row['CPT Projection'])
                self.player_dict[player_name]['Salary'] = int(
                    row['CPT Salary'].replace(',', ''))
                self.player_dict[player_name]['Ceiling'] = float(
                    row['Ceiling']) * 1.5
                self.player_dict[player_name]['StdDev'] = float(
                    row['Ceiling']) * 1.5 - float(row['CPT Projection'])
                self.player_dict[player_name]['Name'] = row['Name']
                self.player_dict[player_name]['Team'] = row['Team']
                self.player_dict[player_name]['Ownership'] = float(
                    row['CPT Own'])
                if self.player_dict[player_name]['Ownership'] == 0:
                    self.player_dict[player_name]['Ownership'] = 1
                self.player_dict[player_name]['Position'] = 'CPT'
                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {player: plp.LpVariable(
            player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts & set randomness amount from config
        self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'],
                                                   (self.player_dict[player]['StdDev'] * self.randomness_amount / 100))
                                  * lp_variables[player] for player in self.player_dict), 'Objective'
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 40000 if self.site == 'dk' else 50000
        self.problem += plp.lpSum(self.player_dict[player]['Salary'] *
                                  lp_variables[player] for player in self.player_dict) <= max_salary
        self.problem += plp.lpSum(self.player_dict[player]['Salary'] *
                                  lp_variables[player] for player in self.player_dict) >= min_salary

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                          for player in group) >= int(limit)

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                          for player in group) <= int(limit)

        for matchup, limit in self.matchup_limits.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) <= int(limit)

        for matchup, limit in self.matchup_at_least.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) >= int(limit)

        # Address team limits
        for team, limit in self.team_limits.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(limit)
        if self.global_team_limit is not None:
            for team in self.team_list:
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(self.global_team_limit)

        if self.site == 'dk':
            # Need exactly 1 CPT
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'CPT' == self.player_dict[player]['Position']) == 1

            # Need exactly 5 FLEX
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'FLEX' == self.player_dict[player]['Position']) == 5

            # Cant roster same player in CPT and FLEX
            for player in self.player_dict:
                self.problem += plp.lpSum(lp_variables[player2]
                                          for player2 in self.player_dict if self.player_dict[player]['Name'] == self.player_dict[player2]['Name']) <= 1
        else:
            print('Not yet supported')
            quit()

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.num_lineups), self.num_lineups))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(
                    v.varValue)).replace('_CPT', '')

            if i % 100 == 0:
                print(i)
            player_names = [v.name.replace(
                '_', ' ').replace('#', '-')[1:] for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names
            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'],
                                                           (self.player_dict[player]['StdDev'] * self.randomness_amount / 100))
                                          * lp_variables[player] for player in self.player_dict), 'Objective'
            else:
                self.problem += plp.lpSum(self.player_dict[player]['Fpts'] * lp_variables[player]
                                          for player in self.player_dict) <= (fpts - 0.01)

    def output(self):
        print('Lineups done generating. Outputting.')

        unique = {}
        for fpts, lineup in self.lineups.items():
            # print(f'fpts: {fpts}, lineup: {lineup}')
            if lineup not in unique.values():
                unique[fpts] = lineup

        if self.num_uniques != 1:
            num_uniq_lineups = plp.OrderedDict(
                sorted(self.lineups.items(), reverse=False, key=lambda t: t[0]))
            self.lineups = {}
            for fpts, lineup in num_uniq_lineups.copy().items():
                temp_lineups = list(num_uniq_lineups.values())
                temp_lineups.remove(lineup)
                use_lineup = True
                for x in temp_lineups:
                    common_players = set(x) & set(lineup)
                    roster_size = 5 if self.site == 'fd' else 6
                    if (roster_size - len(common_players)) < self.num_uniques:
                        use_lineup = False
                        del num_uniq_lineups[fpts]
                        break
                if use_lineup:
                    self.lineups[fpts] = lineup

        self.format_lineups()
        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_optimal_lineups.csv'.format(self.site))

        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'CPT,FLEX,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj, Ceiling, Own. Product,Roster Construction\n')
                for fpts, x in self.lineups.items():
                    home_team_count = 0
                    away_team_count = 0
                    for player in x:
                        if self.player_dict[player.replace('-', '#')]['Team'] == self.home_team:
                            home_team_count += 1
                        else:
                            away_team_count += 1

                    salary = sum(
                        self.player_dict[player.replace('-', '#')]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player.replace('-', '#')]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player.replace('-', '#')]['Ownership'] for player in x])
                    ceil = sum([self.player_dict[player.replace('-', '#')]
                               ['Ceiling'] for player in x])

                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{}'.format(
                        self.player_dict[x[0].replace('-', '#')
                                         ]['Name'], self.player_dict[x[0].replace('-', '#')]['RealID'],
                        self.player_dict[x[1].replace('-', '#')
                                         ]['Name'], self.player_dict[x[1].replace('-', '#')]['RealID'],
                        self.player_dict[x[2].replace('-', '#')
                                         ]['Name'], self.player_dict[x[2].replace('-', '#')]['RealID'],
                        self.player_dict[x[3].replace('-', '#')
                                         ]['Name'], self.player_dict[x[3].replace('-', '#')]['RealID'],
                        self.player_dict[x[4].replace('-', '#')
                                         ]['Name'], self.player_dict[x[4].replace('-', '#')]['RealID'],
                        self.player_dict[x[5].replace('-', '#')
                                         ]['Name'], self.player_dict[x[5].replace('-', '#')]['RealID'],
                        salary, fpts_p, ceil, own_p, f'{home_team_count} {self.home_team} – {away_team_count} {self.away_team}'
                    )
                    f.write('%s\n' % lineup_str)
            else:
                print('Not yet supported')

    def format_lineups(self):
        if self.site == 'dk':
            temp = self.lineups.items()
            self.lineups = {}
            for fpts, lineup in temp:
                finalized = [None] * 6
                curr_idx = 1
                for player in lineup:
                    if 'CPT' in player:
                        finalized[0] = f'_{player}'.replace(' CPT', '_CPT')
                    else:
                        finalized[curr_idx] = f'_{player}'
                        curr_idx = curr_idx + 1

                self.lineups[fpts] = finalized
        else:
            print('Not yet supported')
