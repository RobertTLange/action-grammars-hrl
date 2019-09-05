# -*- coding: utf-8 -*-
import os
import math
import time
import string
import random
import numpy as np
from collections import Counter

def letter_to_action(string_action):
    alphabet = list(string.ascii_lowercase)
    numb_id = list(range(len(alphabet)))
    dic = dict(zip(alphabet, numb_id))
    return dic[string_action]


def action_to_letter(action):
    alphabet = list(string.ascii_lowercase)
    numb_id = list(range(len(alphabet)))
    dic = dict(zip(numb_id, alphabet))
    return dic[action]


class run_grammar():
    """
    Input: path to file with sentence to be used in grammar inference, k for Sequitur
    Output: Grammar object with processed Sequitur/Lexis output
    """
    def __init__(self, string, k=2):
        random_n = str(np.random.randint(1, 100000000))
        self.path_output = "output_" + time.strftime("%Y%m%d-%H%M%S-") + random_n + ".txt"
        self.string = string
        self.k = k

    def run_sequitur(self):
        # Run the Sequitur grammar inferene, outfile results and read them in
        if os.path.exists(self.path_output): os.remove(self.path_output)
        try:
            os.system('echo ' + self.string + ' | ./sequitur -p -k'
                      + str(self.k) + ' >> ' + self.path_output)
            with open(self.path_output) as f:
                self.output = f.read().splitlines()
        except:
            print("Sequitur failed")
        if os.path.exists(self.path_output): os.remove(self.path_output)

    def run_lexis(self):
        # Run the lexis grammar inferene, outfile results and read them in
        os.remove(self.path_output) if os.path.exists(self.path_output) else None
        random_n = str(np.random.randint(1, 100000000))
        path_input =  "input_" + time.strftime("%Y%m%d-%H%M%S-") + random_n + ".txt"
        os.system('echo ' + self.string + ' >> ' + path_input)
        command = 'python2 Lexis.py -m -t c -f e ' + path_input + ' >> ' + self.path_output
        try:
            os.system(command)

            with open(self.path_output) as f:
                self.output = f.read().splitlines()
        except:
            print("Lexis failed")
        if os.path.exists(path_input): os.remove(path_input)
        if os.path.exists(self.path_output): os.remove(self.path_output)

    def clean_sequitur(self):
        # Extract non-terminal symbols and corresponding productions
        nonterminals = []
        productions = []

        for line in self.output:
            try:
                production = line.split(" -> ", 1)[1]
                nonterms = line.split(" -> ", 1)[0]
                nonterminals.append(nonterms)
                productions.append(production)
            except:
                pass

        # Rename all nonterminals by numbers and remove spaces
        rename_list = range(1, len(nonterminals))
        rename_dict = dict(zip(nonterminals, rename_list))

        productions[0] = productions[0].replace("\\n", "")
        # Add awkward "-" to be able to differentiate when working with >9 prod
        for i, prod in enumerate(productions):
            productions[i] = productions[i].replace(" ", "-")
            productions[i] = productions[i][:-1]
        productions[0] = productions[0][:-1]
        # Recursively flatten the production rules by plugging productions in
        flat_productions = productions[:]
        while any(any(char.isdigit() for char in prod) for prod in flat_productions):
            for numb in reversed(rename_list):
                for i, prod in enumerate(flat_productions):
                    if str(numb) in prod:
                        flat_productions[i] = flat_productions[i].replace(str(numb),
                                                             flat_productions[numb])
        # Construct terminals as set diff of all unique symbols and nonterms
        splitted_rules = [list(prod) for prod in productions]
        uniques = list(set([item for subl in splitted_rules for item in subl]))

        terminals = list(set(uniques) - set(rename_list))

        for i, prod in enumerate(flat_productions):
            flat_productions[i] = flat_productions[i].replace("-", "")

        # Collect outputs
        self.uniques = uniques
        self.terminals = terminals
        self.nonterminals = rename_list
        self.productions = productions
        self.flat_productions = flat_productions

        # S - encoded sequence, N - production rules
        self.S = self.productions[0]
        self.N = self.productions[1:]
        # logging.info("Successfully extracted symbolic relations")

    def clean_lexis(self):
        # Extract non-terminal symbols and corresponding productions
        nonterminals = []
        productions = []

        for line in self.output:
            try:
                production = line.split(" -> ", 1)[1]
                nonterms = line.split(" -> ", 1)[0]
                nonterminals.append(nonterms)
                productions.append(production)
            except:
                pass

        print(nonterminals)
        print(productions)
        rename_list = range(1, len(nonterminals))
        rename_dict = dict(zip(nonterminals, rename_list))

        for i, prod in enumerate(productions):
            prod_temp = prod.split()
            for key in rename_dict.keys():
                for j in range(len(prod_temp)):
                    if key == prod_temp[j]:
                        prod_temp[j] = str(rename_dict[key])
            productions[i] = " ".join(prod_temp)

        # Rename all nonterminals by numbers and remove spaces
        rename_list = range(1, len(nonterminals))
        rename_dict = dict(zip(nonterminals, rename_list))

        productions[0] = productions[0].replace("\\n", "")
        # Add awkward "-" to be able to differentiate when working with >9 prod
        for i, prod in enumerate(productions):
            productions[i] = productions[i].replace(" ", "-")
            productions[i] = productions[i][:-1]
        productions[0] = productions[0][:-1]
        # Recursively flatten the production rules by plugging productions in
        flat_productions = productions[:]
        while any(any(char.isdigit() for char in prod) for prod in flat_productions):
            for numb in reversed(rename_list):
                for i, prod in enumerate(flat_productions):
                    if str(numb) in prod:
                        flat_productions[i] = flat_productions[i].replace(str(numb),
                                                             flat_productions[numb])
        # Construct terminals as set diff of all unique symbols and nonterms
        splitted_rules = [list(prod) for prod in productions]
        uniques = list(set([item for subl in splitted_rules for item in subl]))

        terminals = list(set(uniques) - set(rename_list))

        for i, prod in enumerate(flat_productions):
            flat_productions[i] = flat_productions[i].replace("-", "")

        # Collect outputs
        self.uniques = uniques
        self.terminals = terminals
        self.nonterminals = rename_list
        self.productions = productions
        self.flat_productions = flat_productions

        # S - encoded sequence, N - production rules
        self.S = self.productions[0]
        self.N = self.productions[1:]
        # logging.info("Successfully extracted symbolic relations")

    def get_compression_stats(self, print_out=False):
        # Obtain compression info (i.e. ration of encoded and original string)
        lengths = []

        init_seq = self.flat_productions[0]
        lengths.append(len(init_seq))
        lengths.append(len(self.S))
        self.lengths = lengths
        self.length_ratio = float(lengths[1])/lengths[0]

        def shannon_entropy(string):
            prob = [float(string.count(c))/len(string) for c in dict.fromkeys(string)]
            entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
            return entropy

        comp_ratio = float(len(init_seq))/len(self.S)
        shannon_pre = shannon_entropy(init_seq)
        shannon_post = shannon_entropy(self.S)

        if print_out:
            print("Compression Ratio:", comp_ratio)
            print("Pre-Compression Shannon Entropy:", shannon_pre)
            print("Post-Compression Shannon Entropy:", shannon_post)

        stats = [comp_ratio, shannon_pre, shannon_post]
        return stats


def get_macros(NUM_MACROS, SENTENCE, NUM_PRIMITIVES, GRAMMAR_DIR, k=2, g_type="sequitur"):
    original_dir = os.getcwd()
    primitives =  list(string.ascii_lowercase)[:NUM_PRIMITIVES]

    Grammar = run_grammar(SENTENCE, k)

    os.chdir(GRAMMAR_DIR)
    if g_type == "sequitur":
        Grammar.run_sequitur()
        Grammar.clean_sequitur()
    elif g_type == "lexis":
        Grammar.run_lexis()
        Grammar.clean_lexis()

    stats = Grammar.get_compression_stats()

    temp_S = Grammar.S.split("-")
    occ = dict(Counter(temp_S))

    for key in primitives:
        try:
            del occ[key]
        except:
            continue

    # If not all macros shall be returned - sort them by occurence -
    # only return most frequenlty used in encoding of string
    if NUM_MACROS != "all":
        sorted_occ = sorted(occ.items(), key=lambda x: x[1], reverse=True)
        sorted_occ = sorted_occ[0:NUM_MACROS]
        counts = [int(m[1]) for m in sorted_occ]
        sorted_macros = [int(m[0]) for m in sorted_occ]
        macros = [Grammar.flat_productions[i] for i in sorted_macros]
    else:
        counts = list(occ.values())
        macros = Grammar.flat_productions[1:]

    os.chdir(original_dir)
    return macros, counts, stats
