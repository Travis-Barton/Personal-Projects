#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:18:22 2018

@author: travisbarton
"""

import bs4, requests
exampleFile = open('example.html')
exampleSoup = bs4.BeautifulSoup(exampleFile.read())
elems = exampleSoup.select('#author')
print type(elems)
print len(elems)
print type(elems[0])
print elems[0].getText()
print str(elems[0])
print elems[0].attrs
pElms = 
                           