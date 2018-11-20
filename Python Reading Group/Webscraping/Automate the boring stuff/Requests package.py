#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:05:57 2018

@author: travisbarton
"""

import requests
res = requests.get('http://www.gutenberg.org/cache/epub/1112/pg1112.txt')
type(res)
res.status_code == requests.codes.ok

len(res.text)

print(res.text[:250])



try:
    fake.fake
except:
    print "ERROR"
    
    
res = requests.get('http://www.gutenberg.org/cache/epub/1112/pg1112.txt')
try:
    res.raise_for_status()
except Exception as exc:
    print "There was an error: %s'" % (exc)
playfile = open('RomeoAndJuliet.txt', 'wb')
for chunk in res.iter_content(10):
    playfile.write(chunk)


playfile.close()
