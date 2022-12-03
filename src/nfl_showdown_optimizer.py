import json
import csv
import os
import re
import pulp as plp


class NFL_Showdown_Optimizer:
    problem = None
    config = None
    output_dir = None
    num_lineups = None
    player_dict = {}
    max_salary = None
    lineups = []

    def __init__(self):
        self.problem = plp.LpProblem('NFL', plp.LpMaximize)
        self.load_config()
        self.load_rules()
        self.load_projections(self.config['projection_path'])
        self.output_dir = self.config['output_dir']
        self.num_lineups = self.config['num_lineups']

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # projection
                self.player_projections[row['Position']
                                        ][row['Name']] = float(row['Fpts'])
                # salary
                self.player_salaries[row['Position']][row['Name']] = int(
                    row['Salary'].replace(',', ''))

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])

    def optimize(self):
        print(self)
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.

    def output(self):
        div = '---------------------------------------\n'
        print('Variables:\n')
        score = str(self.problem.objective)
        constraints = [str(const)
                       for const in self.problem.constraints.values()]
        for v in self.problem.variables():
            score = score.replace(v.name, str(v.varValue))
            constraints = [const.replace(v.name, str(v.varValue))
                           for const in constraints]
            if v.varValue != 0:
                print(v.name, '=', v.varValue)
        print(div)
        print('Constraints:')
        for constraint in constraints:
            constraint_pretty = ' + '.join(
                re.findall('[0-9\.]*\*1.0', constraint))
            if constraint_pretty != '':
                print('{} = {}'.format(constraint_pretty, eval(constraint_pretty)))
        print(div)
        print('Score:')
        score_pretty = ' + '.join(re.findall('[0-9\.]+\*1.0', score))
        print('{} = {}'.format(score_pretty, eval(score)))
        # with open(self.output_dir, 'w') as f:
        #     f.write('QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts,Salary,Ownership\n')
        #     for x in final:
        #         fpts = sum(float(projection_dict[player]['Fpts']) for player in x)
        #         salary = sum(int(projection_dict[player]['Salary'].replace(',','')) for player in x)
        #         own = sum(float(ownership_dict[player]) for player in x)
        #         lineup_str = '{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        #             x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],fpts,salary,own
        #         )
        #         f.write('%s\n' % lineup_str)
