import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp


class NFL_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    gen_lineup_list = []
    roster_construction = []
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
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    use_double_te = True

    def __init__(self, site, field_size, num_iterations, use_contest_data, use_lineup_input,
                 match_lineup_input_to_field_size):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.match_lineup_input_to_field_size = match_lineup_input_to_field_size
        self.load_config()
        self.load_rules()
        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)
        ownership_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        self.load_ownership(ownership_path)

        boom_bust_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        if site == 'dk':
            self.roster_construction = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
            self.salary = 50000
        elif site == 'fd':
            self.roster_construction = [
                'QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
            self.salary = 60000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(os.path.dirname(
                __file__), '../{}_data/{}'.format(site, self.config['contest_structure_path']))
            self.load_contest_data(contest_path)
            print('Contest payout structure loaded.')
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0
        self.num_iterations = int(num_iterations)
        print(self.player_dict)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
            self.generate_field_lineups()

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `nba_optimizer.py`
    def get_optimal(self):
        problem = plp.LpProblem('NFL', plp.LpMaximize)
        lp_variables = {player: plp.LpVariable(
            player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        problem += plp.lpSum(self.player_dict[player]['Fpts'] * lp_variables[player]
                             for player in self.player_dict), 'Objective'

        # Set the salary constraints
        problem += plp.lpSum(self.player_dict[player]['Salary'] * lp_variables[player]
                             for player in self.player_dict) <= self.salary

        # Need exactly 1 QB
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'QB' in self.player_dict[player]['Position']) == 1

        # Need at least 2 RB, up to 3 if using FLEX
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'RB' in self.player_dict[player]['Position']) >= 2
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'RB' in self.player_dict[player]['Position']) <= 3

        # Need at least 3 WR, up to 4 if using FLEX
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'WR' in self.player_dict[player]['Position']) >= 3
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'WR' in self.player_dict[player]['Position']) <= 4

        # Need at least 1 TE, up to 2 if using FLEX
        if self.use_double_te:
            problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if
                                      'TE' in self.player_dict[player]['Position']) >= 1
            problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if
                                      'TE' in self.player_dict[player]['Position']) <= 2
        else:
            problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if
                                      'TE' in self.player_dict[player]['Position']) == 1

        # Need exactly 1 DST
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict if
                                  'DST' in self.player_dict[player]['Position']) == 1

        # Can only roster 9 total players
        problem += plp.lpSum(lp_variables[player]
                                  for player in self.player_dict) == 9

        # Crunch!
        try:
            #problem.writeLP('./problemLP.lp')
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print('Infeasibility reached')

        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[player_name]['ID'] = int(row['ID'])
                    else:
                        self.player_dict[player_name]['ID'] = row['Id']

    def load_contest_data(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row['Field Size'])
                if self.entry_fee is None:
                    self.entry_fee = float(row['Entry Fee'])
                # multi-position payouts
                if '-' in row['Place']:
                    indices = row['Place'].split('-')
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row['Payout'].split('.')[0].replace(',', ''))
                # single-position payouts
                else:
                    if int(row['Place']) >= self.field_size:
                        break
                    self.payout_structure[int(
                        row['Place']) - 1] = float(row['Payout'].split('.')[0].replace(',', ''))
        # print(self.payout_structure)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json'), encoding='utf-8-sig') as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if float(row['Fpts']) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {'Fpts': 0, 'Position': [
                ], 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ceiling': 0, 'Ownership': 0.1, 'In Lineup': False}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(
                    row['Salary'].replace(',', ''))

                self.player_dict[player_name]['Team'] = row['Team']

                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])

                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                if self.site == 'dk':
                    self.player_dict[player_name]['Position'] = [
                        pos for pos in row['Position'].split('/')]

                    if 'DST' in self.player_dict[player_name]['Position']:
                        self.player_dict[player_name]['Position'].append('')
                    else:
                        self.player_dict[player_name]['Position'].append('FLEX')


                elif self.site == 'fd':
                    self.player_dict[player_name]['Position'] = [
                        pos for pos in row['Position'].split('/')]

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row["Name"].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(
                        row['Own%'])

    # Load standard deviations
    def load_boom_bust(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(
                        row['Std Dev'])
                    self.player_dict[player_name]['Ceiling'] = float(
                        row['Ceiling'])

    def remap(self, fieldnames):
        return ['PG', 'PG2', 'SG', 'SG2', 'SF', 'SF2', 'PF', 'PF2', 'C']

    def load_lineups_from_file(self):
        print('loading lineups')
        i = 0
        path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(self.site, 'tournament_lineups.csv'))
        with open(path) as file:
            if self.site == 'dk':
                reader = csv.DictReader(file)
                for row in reader:
                    if i == self.field_size:
                        break
                    lineup = [row['PG'].split('(')[0][:-1].replace('-', '#'),
                              row['SG'].split('(')[0][:-1].replace('-', '#'),
                              row['SF'].split(
                                  '(')[0][:-1].replace('-', '#'), row['PF'].split('(')[0][:-1].replace('-', '#'),
                              row['C'].split(
                                  '(')[0][:-1].replace('-', '#'), row['G'].split('(')[0][:-1].replace('-', '#'),
                              row['F'].split('(')[0][:-1].replace('-', '#'),
                              row['UTIL'].split('(')[0][:-1].replace('-', '#')]
                    # storing if this lineup was made by an optimizer or with the generation process in this script
                    self.field_lineups[i] = {
                        'Lineup': lineup, 'Wins': 0, 'Top10': 0, 'ROI': 0, 'Cashes': 0, 'Type': 'opto'}
                    i += 1
            else:
                reader = csv.reader(file)
                fieldnames = self.remap(next(reader))
                for row in reader:
                    row = dict(zip(fieldnames, row))
                    if i == self.field_size:
                        break
                    lineup = [row['PG'].split(':')[1].replace('-', '#'), row['PG2'].split(':')[1].replace('-', '#'),
                              row['SG'].split(':')[1].replace(
                                  '-', '#'), row['SG2'].split(':')[1].replace('-', '#'),
                              row['SF'].split(':')[1].replace(
                                  '-', '#'), row['SF2'].split(':')[1].replace('-', '#'),
                              row['PF'].split(':')[1].replace(
                                  '-', '#'), row['PF2'].split(':')[1].replace('-', '#'),
                              row['C'].split(':')[1].replace('-', '#')]
                    self.field_lineups[i] = {
                        'Lineup': lineup, 'Wins': 0, 'Top10': 0, 'ROI': 0, 'Cashes': 0, 'Type': 'opto'}
                    i += 1

    @staticmethod
    def generate_lineups(lu_num, names, in_lineup, pos_matrix, ownership, salary_floor, salary_ceiling, optimal_score,
                         salaries, projections, max_pct_off_optimal):
        # new random seed for each lineup (without this there is a ton of dupes)
        np.random.seed(lu_num)
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        while reject:
            salary = 0
            proj = 0
            if sum(in_lineup) != 0:
                in_lineup.fill(0)
            lineup = []
            for pos in pos_matrix:
                # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                valid_players = np.where((pos > 0) & (in_lineup == 0))
                # grab names of players eligible
                plyr_list = names[valid_players]
                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()
                choice = np.random.choice(a=plyr_list, p=prob_list)
                choice_idx = np.where(names == choice)[0]
                lineup.append(choice)
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]
            # Must have a reasonable salary
            if (salary >= salary_floor and salary <= salary_ceiling):
                # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                reasonable_projection = optimal_score - \
                                        (max_pct_off_optimal * optimal_score)
                if proj >= reasonable_projection:
                    reject = False
                    lus[lu_num] = {
                        'Lineup': lineup, 'Wins': 0, 'Top10': 0, 'ROI': 0, 'Cashes': 0, 'Type': 'generated'}
        return lus

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print('supplied lineups >= contest field size. only retrieving the first ' + str(
                self.field_size) + ' lineups')
        else:
            print('Generating ' + str(diff) + ' lineups.')
            names = list(self.player_dict.keys())
            in_lineup = np.zeros(shape=len(names))
            i = 0
            ownership = np.array([self.player_dict[player_name]['Ownership'] / 100 for player_name in names])
            salaries = np.array([self.player_dict[player_name]['Salary'] for player_name in names])
            projections = np.array([self.player_dict[player_name]['Fpts'] for player_name in names])
            positions = []
            for pos in self.roster_construction:
                pos_list = []
                own = []
                for player_name in names:
                    if pos in self.player_dict[player_name]['Position']:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                i += 1
                positions.append(np.array(pos_list))
            pos_matrix = np.array(positions)
            names = np.array(names)
            optimal_score = self.optimal_score
            salary_floor = self.min_lineup_salary  # anecdotally made the most sense when looking at previous contests
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            problems = []
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                i, names, in_lineup, pos_matrix, ownership, salary_floor, salary_ceiling, optimal_score, salaries,
                projections, max_pct_off_optimal)
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print('number of running processes =',
                      pool.__dict__['_processes']
                      if (pool.__dict__['_state']).upper() == 'RUN'
                      else None
                      )
                pool.close()
                pool.join()
            if len(self.field_lineups) == 0:
                new_keys = list(range(0, self.field_size))
            else:
                new_keys = list(range(max(self.field_lineups.keys()) + 1, self.field_size))
            nk = new_keys[0]
            for i, o in enumerate(output):
                if nk in self.field_lineups.keys():
                    print('bad lineups dict, please check dk_data files')
                self.field_lineups[nk] = o[i]
                nk += 1
            end_time = time.time()
            print('lineups took ' + str(end_time - start_time) + ' seconds')
            print(str(diff) + ' field lineups successfully generated')

    def run_tournament_simulation(self):
        print('Running ' + str(self.num_iterations) + ' simulations')
        start_time = time.time()
        temp_fpts_dict = {p: np.random.normal(
            s['Fpts'], s['StdDev'] * self.randomness_amount / 100, size=self.num_iterations) for p, s in
            self.player_dict.items()}
        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(self.field_size, self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(shape=self.field_size -
                                len(payout_array), fill_value=-self.entry_fee)
        payout_array = np.concatenate((payout_array, l_array))
        for index, values in self.field_lineups.items():
            fpts_sim = sum([temp_fpts_dict[player]
                            for player in values['Lineup']])
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim
        ranks = np.argsort(fpts_array, axis=0)[::-1]
        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        t10, t10_counts = np.unique(ranks[0:9:], return_counts=True)
        roi = payout_array[np.argsort(ranks, axis=0)].sum(axis=1)
        # summing up ach lineup, probably a way to v)ectorize this too (maybe just turning the field dict into an array too)
        for idx in self.field_lineups.keys():
            # Winning
            if idx in wins:
                self.field_lineups[idx]['Wins'] += win_counts[np.where(
                    wins == idx)][0]
            # Top 10
            if idx in t10:
                self.field_lineups[idx]['Top10'] += t10_counts[np.where(
                    t10 == idx)][0]
            # can't figure out how to get roi for each lineup index without iterating and iterating is slow
            if self.use_contest_data:
                #    self.field_lineups[idx]['ROI'] -= (loss_counts[np.where(losses==idx)][0])*self.entry_fee
                self.field_lineups[idx]['ROI'] += roi[idx]
        end_time = time.time()
        diff = end_time - start_time
        print(str(self.num_iterations) +
              ' tournament simulations finished in ' + str(diff) + 'seconds. Outputting.')

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            salary = sum(self.player_dict[player]['Salary']
                         for player in x['Lineup'])
            fpts_p = sum(self.player_dict[player]['Fpts']
                         for player in x['Lineup'])
            ceil_p = sum(self.player_dict[player]['Ceiling']
                         for player in x['Lineup'])
            own_p = np.prod(
                [self.player_dict[player]['Ownership'] / 100.0 for player in x['Lineup']])
            win_p = round(x['Wins'] / self.num_iterations * 100, 2)
            top10_p = round(x['Top10'] / self.num_iterations * 100, 2)
            cash_p = round(x['Cashes'] / self.num_iterations * 100, 2)
            lu_type = x['Type']
            if self.site == 'dk':
                if self.use_contest_data:
                    roi_p = round(x['ROI'] / self.entry_fee /
                                  self.num_iterations * 100, 2)
                    roi_round = round(x['ROI'] / self.num_iterations, 2)
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{}%,{},${},{}'.format(
                        x['Lineup'][0].replace(
                            '#', '-'), self.player_dict[x['Lineup'][0].replace('-', '#')]['ID'],
                        x['Lineup'][1].replace(
                            '#', '-'), self.player_dict[x['Lineup'][1].replace('-', '#')]['ID'],
                        x['Lineup'][2].replace(
                            '#', '-'), self.player_dict[x['Lineup'][2].replace('-', '#')]['ID'],
                        x['Lineup'][3].replace(
                            '#', '-'), self.player_dict[x['Lineup'][3].replace('-', '#')]['ID'],
                        x['Lineup'][4].replace(
                            '#', '-'), self.player_dict[x['Lineup'][4].replace('-', '#')]['ID'],
                        x['Lineup'][5].replace(
                            '#', '-'), self.player_dict[x['Lineup'][5].replace('-', '#')]['ID'],
                        x['Lineup'][6].replace(
                            '#', '-'), self.player_dict[x['Lineup'][6].replace('-', '#')]['ID'],
                        x['Lineup'][7].replace(
                            '#', '-'), self.player_dict[x['Lineup'][7].replace('-', '#')]['ID'],
                        fpts_p, ceil_p, salary, win_p, top10_p, roi_p, own_p, roi_round, lu_type
                    )
                else:
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{},{}%,{}'.format(
                        x['Lineup'][0].replace(
                            '#', '-'), self.player_dict[x['Lineup'][0]]['ID'],
                        x['Lineup'][1].replace(
                            '#', '-'), self.player_dict[x['Lineup'][1]]['ID'],
                        x['Lineup'][2].replace(
                            '#', '-'), self.player_dict[x['Lineup'][2]]['ID'],
                        x['Lineup'][3].replace(
                            '#', '-'), self.player_dict[x['Lineup'][3]]['ID'],
                        x['Lineup'][4].replace(
                            '#', '-'), self.player_dict[x['Lineup'][4]]['ID'],
                        x['Lineup'][5].replace(
                            '#', '-'), self.player_dict[x['Lineup'][5]]['ID'],
                        x['Lineup'][6].replace(
                            '#', '-'), self.player_dict[x['Lineup'][6]]['ID'],
                        x['Lineup'][7].replace(
                            '#', '-'), self.player_dict[x['Lineup'][7]]['ID'],
                        fpts_p, ceil_p, salary, win_p, top10_p, own_p, cash_p, lu_type
                    )
            elif self.site == 'fd':
                if self.use_contest_data:
                    roi_p = round(x['ROI'] / self.entry_fee /
                                  self.num_iterations * 100, 2)
                    roi_round = round(x['ROI'] / self.num_iterations, 2)
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{}%,{},${},{}'.format(
                        self.player_dict[x['Lineup'][0].replace(
                            '-', '#')]['ID'], x['Lineup'][0].replace('#', '-'),
                        self.player_dict[x['Lineup'][1].replace(
                            '-', '#')]['ID'], x['Lineup'][1].replace('#', '-'),
                        self.player_dict[x['Lineup'][2].replace(
                            '-', '#')]['ID'], x['Lineup'][2].replace('#', '-'),
                        self.player_dict[x['Lineup'][3].replace(
                            '-', '#')]['ID'], x['Lineup'][3].replace('#', '-'),
                        self.player_dict[x['Lineup'][4].replace(
                            '-', '#')]['ID'], x['Lineup'][4].replace('#', '-'),
                        self.player_dict[x['Lineup'][5].replace(
                            '-', '#')]['ID'], x['Lineup'][5].replace('#', '-'),
                        self.player_dict[x['Lineup'][6].replace(
                            '-', '#')]['ID'], x['Lineup'][6].replace('#', '-'),
                        self.player_dict[x['Lineup'][7].replace(
                            '-', '#')]['ID'], x['Lineup'][7].replace('#', '-'),
                        self.player_dict[x['Lineup'][8].replace(
                            '-', '#')]['ID'], x['Lineup'][8].replace('#', '-'),
                        fpts_p, ceil_p, salary, win_p, top10_p, roi_p, own_p, roi_round, lu_type
                    )
                else:
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{},{}%,{}'.format(
                        self.player_dict[x['Lineup'][0].replace(
                            '-', '#')]['ID'], x['Lineup'][0].replace('#', '-'),
                        self.player_dict[x['Lineup'][1].replace(
                            '-', '#')]['ID'], x['Lineup'][1].replace('#', '-'),
                        self.player_dict[x['Lineup'][2].replace(
                            '-', '#')]['ID'], x['Lineup'][2].replace('#', '-'),
                        self.player_dict[x['Lineup'][3].replace(
                            '-', '#')]['ID'], x['Lineup'][3].replace('#', '-'),
                        self.player_dict[x['Lineup'][4].replace(
                            '-', '#')]['ID'], x['Lineup'][4].replace('#', '-'),
                        self.player_dict[x['Lineup'][5].replace(
                            '-', '#')]['ID'], x['Lineup'][5].replace('#', '-'),
                        self.player_dict[x['Lineup'][6].replace(
                            '-', '#')]['ID'], x['Lineup'][6].replace('#', '-'),
                        self.player_dict[x['Lineup'][7].replace(
                            '-', '#')]['ID'], x['Lineup'][7].replace('#', '-'),
                        self.player_dict[x['Lineup'][8].replace(
                            '-', '#')]['ID'], x['Lineup'][8].replace('#', '-'),
                        fpts_p, ceil_p, salary, win_p, top10_p, own_p, cash_p, lu_type
                    )
            unique[index] = lineup_str

        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_gpp_sim_lineups_{}_{}.csv'.format(self.site, self.field_size, self.num_iterations))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                if self.use_contest_data:
                    f.write(
                        'PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product, Avg. Return,Type\n')
                else:
                    f.write(
                        'PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Cash %,Type\n')
            elif self.site == 'fd':
                if self.use_contest_data:
                    f.write(
                        'PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product, Avg. Return,Type\n')
                else:
                    f.write(
                        'PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Cash %,Type\n')

            for fpts, lineup_str in unique.items():
                f.write('%s\n' % lineup_str)

        out_path = os.path.join(os.path.dirname(
            __file__),
            '../output/{}_gpp_sim_player_exposure_{}_{}.csv'.format(self.site, self.field_size, self.num_iterations))
        with open(out_path, 'w') as f:
            f.write('Player,Win%,Top10%,Sim. Own%,Proj. Own%,Avg. Return\n')
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val['Lineup']:
                    if player not in unique_players:
                        unique_players[player] = {
                            'Wins': val['Wins'], 'Top10': val['Top10'], 'In': 1, 'ROI': val['ROI']}
                    else:
                        unique_players[player]['Wins'] = unique_players[player]['Wins'] + val['Wins']
                        unique_players[player]['Top10'] = unique_players[player]['Top10'] + val['Top10']
                        unique_players[player]['In'] = unique_players[player]['In'] + 1
                        unique_players[player]['ROI'] = unique_players[player]['ROI'] + val['ROI']

            for player, data in unique_players.items():
                field_p = round(data['In'] / self.field_size * 100, 2)
                win_p = round(data['Wins'] / self.num_iterations * 100, 2)
                top10_p = round(
                    data['Top10'] / self.num_iterations / 10 * 100, 2)
                roi_p = round(data['ROI'] / data['In'] / self.num_iterations, 2)
                proj_own = self.player_dict[player]['Ownership']
                f.write('{},{}%,{}%,{}%,{}%,${}\n'.format(player.replace(
                    '#', '-'), win_p, top10_p, field_p, proj_own, roi_p))