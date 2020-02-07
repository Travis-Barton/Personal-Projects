# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tweepy
import tkinter
consumer_key = 'QhV5mrXMuX9z54Z2tYdHC1I3U'
consumer_secret = '2ouNVyhrkcy4LQoWTVi2iuTdcPAphPnsjmJAcXGx98P0oVp7oM'
access_token = '107270162-toI6Q1a3z85xf01QfzmCSyHxkMx3z96PbtGyL78F'
access_token_secret = '8n9onMf9c7k7yFEM42qfzfvTAKwAJ9tEgJp0p6Ldi4Dlu'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
user = api.me()
print (user.name)
api.update_status(status = 'Trying to update my status with python!')
api.update_status(status = 'Ayyyye it worked, now to build bots')
