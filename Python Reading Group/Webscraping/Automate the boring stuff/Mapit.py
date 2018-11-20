#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 18:51:58 2018

@author: travisbarton
"""
import webbrowser as wb
import sys, pyperclip, requests
#wb.open('http://www.reddit.com/')

if len(sys.argv) > 1:
    adress = " ".join(sys.argv[1:])
else:
    address = pyperclip.paste()
    
print address

wb.open('https://www.google.com/maps/place/' + address)

