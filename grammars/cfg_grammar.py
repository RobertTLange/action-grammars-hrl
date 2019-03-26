# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import math
import time
import string
import random
import numpy as np
from collections import Counter
from utils import *

original_dir = os.getcwd()
base_dir =  original_dir + "/grammars"
trace_dir = base_dir + "/traces/"
seq_dir = base_dir + "/sequitur/"
lexis_dir = base_dir + "/Lexis/"

class run_grammar():
    """
    Input: path to file with sentence to be used in grammar inference, k for Sequitur
    Output: Grammar object with processed Sequitur/Lexis output
    """
    def __init__(self, path_string, k=2):
        self.path_string = path_string
        random_n = str(random.randint(1, 100000000))
        self.path_output = "output_" + time.strftime("%Y%m%d-%H%M%S-") + random_n + ".txt"
        with open(self.path_string) as f:
            string = f.read().splitlines()

        self.string = string[0]
        self.k = k

    def run_sequitur(self):
        # Run the Sequitur grammar inferene, outfile results and read them in
        os.remove(self.path_output) if os.path.exists(self.path_output) else None
        try:
            os.system('echo ' + self.string + ' | ./sequitur -p -k'
                      + str(self.k) + ' >> ' + self.path_output)

            with open(self.path_output) as f:
                self.output = f.read().splitlines()

            self.g_type = "sequitur"
        except:
            print("Sequitur failed")
        os.remove(self.path_output) if os.path.exists(self.path_output) else None

    def run_lexis(self):
        # Run the lexis grammar inferene, outfile results and read them in
        os.remove(self.path_output) if os.path.exists(self.path_output) else None
        try:
            command = 'python Lexis.py -m -t c -f i ' + self.path_string + ' >> ' + self.path_output
            os.system(command)

            with open(self.path_output) as f:
                self.output = f.read().splitlines()
            self.g_type = "lexis"
        except:
            print("Lexis failed")
        os.remove(self.path_output) if os.path.exists(self.path_output) else None

    def clean_output(self):
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

        if self.g_type == "lexis":
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
        print(productions)
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

        logging.info("Successfully computed compression statistics")

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

        return comp_ratio, shannon_pre, shannon_post


def get_macros_from_traces(env, no_macros, action_list, g_type="sequitur", k=2):
    num_primitives = env.action_space.n
    encoded_seqs = encode_actions(action_list, num_primitives)

    # Join strings - looping over traces does not work!
    full_sentence = ".".join(encoded_seqs)

    # for sentence in encoded_seqs:
    #     macros_temp = get_macros(no_macros, sentence, num_primitives, g_type, k)
    #     all_macros.extend(macros_temp)

    macros_temp = get_macros(no_macros, full_sentence, num_primitives, g_type, k)
    clean_macros =  [x for x in macros_temp if "." not in x]
    return list(set(clean_macros))


def get_macros(no_macros, sentence, num_primitives, g_type="sequitur", k=2):
    primitives =  list(string.ascii_lowercase)[:num_primitives]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    random_n = str(random.randint(1, 100000000))
    temp = "temp_" + timestr + str(random_n) + ".txt"

    with open(trace_dir + temp, "w") as myfile:
        myfile.write(sentence)

    Grammar = run_grammar(trace_dir + temp, k)
    if g_type == "sequitur":
        os.chdir(seq_dir)
        Grammar.run_sequitur()
    if g_type == "lexis":
        os.chdir(lexis_dir)
        Grammar.run_lexis()

    Grammar.clean_output()

    temp_S = Grammar.S.split("-")
    occ = dict(Counter(temp_S))
    for key in primitives:
        try:
            del occ[key]
        except:
            continue

    if no_macros != "all":
        sorted_occ = sorted(occ.items(), key=lambda x: x[1], reverse=True)
        sorted_occ = sorted_occ[0:no_macros]
        sorted_macros = [int(m[0]) for m in sorted_occ]
        macros = [Grammar.flat_productions[i] for i in sorted_macros]
    else:
        macros = Grammar.flat_productions[1:]

    os.remove(trace_dir + temp) if os.path.exists(trace_dir + temp) else None
    os.chdir(original_dir)
    return macros
