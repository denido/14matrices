# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:38:24 2019

@author: OEM
"""

import argparse
import yaml
import modules.class_fourteen_matrices



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",help="Config file")
    parser.add_argument("--mode",help="run mode")
    args = parser.parse_args()
    
    with open(args.config,"r") as yamlfile:
        cfg = yaml.load(yamlfile)
        
    system = modules.class_fourteen_matrices.fourteen_matrices(cfg)
    if args.mode == "run":
        system.NEGF()
        system.plot()
    