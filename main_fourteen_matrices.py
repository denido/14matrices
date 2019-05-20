# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:38:24 2019

@author: OEM
"""

import argparse
import numpy as np
import os
#import subprocess
import sys
import yaml
import class_fourteen_matrices
#import S1_selfenergy
#import CNTFET


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",help="Config file")
    parser.add_argument("--mode",help="run mode")
    args = parser.parse_args()
    
    
    
    with open(args.config,"r") as yamlfile:
        cfg = yaml.load(yamlfile)
        
    system = class_fourteen_matrices.fourteen_matrices(cfg)
    
    if args.mode == "run":
        system.NEGF()
        
    if args.mode == "plot":
        system.plot()
    