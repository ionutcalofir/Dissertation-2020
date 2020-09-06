"""
Get all results from https://understat.com/
"""

import os
import asyncio
import json
import aiohttp
from understat import Understat

MATCHES_DIR = 'matches'
os.makedirs(MATCHES_DIR, exist_ok=True)

async def main(league, year):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        matches = await understat.get_league_results(
            league,
            year
        )

        json.dump(matches, open(os.path.join(MATCHES_DIR, 'matches_{}_{}.json'.format(league, year)), 'w'))

loop = asyncio.get_event_loop()

leagues = ['epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
years = ['2014', '2015', '2016', '2017', '2018', '2019']

for league in leagues:
    for year in years:
        print('Process {} {}'.format(league, year))
        loop.run_until_complete(main(league, year))