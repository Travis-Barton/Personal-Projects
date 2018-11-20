#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:47:55 2018

@author: travisbarton
"""

import sys, requests, bs4
import webbrowser as wb

print "Googling..."
res = requests.get('https://google.com/search?q=' + ''.join(sys.argv[1:]))
res.raise_for_status()

soup = bs4.BeautifulSoup(res.text)
linkElems = soup.select('.r a')
numOpen = min(3, len(linkElems))
for i in range(numOpen):
    wb.open('http://google.com' + linkElems[i].get('href'))
