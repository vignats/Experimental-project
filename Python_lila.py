# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:31:54 2022

@author: lilac
"""

import re
with open("breves_doublons.txt","r") as file:
    for lignes,i enumerate (file):
        while   
        regex=re.compile("([a-z]+)\s \\1")
        new=regex.sub("", lignes)
        print(new)