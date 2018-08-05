import random

import cassiopeia as cass

cass.set_riot_api_key("RGAPI-378d7c50-ec70-473a-a8b5-4a05a97847c3")
cass.set_default_region("NA")

summoner = cass.Summoner(name = "Love2Learn", id = 47596304)
good_with = summoner.champion_masteries.filter(
        lambda cm: cm.level>=6)
print([cm.champion.name for cm in good_with])
