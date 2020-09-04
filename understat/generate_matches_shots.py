import os
import asyncio
import json
import aiohttp
from understat import Understat

MATCHES_DIR = 'matches'
MATCHES_SHOTS_DIR = 'matches_shots'
os.makedirs(MATCHES_SHOTS_DIR, exist_ok=True)

matches_shots = None
async def main(match_id):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        try:
            match_shots = await understat.get_match_shots(
                match_id,
            )
        except:
            print('Match {} could not be retrieved!'.format(match_id))
            match_shots = None

        matches_shots.append(match_shots)

loop = asyncio.get_event_loop()

leagues = ['epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
years = ['2014', '2015', '2016', '2017', '2018', '2019']

for league in leagues:
    for year in years:
        print('Process {} {}'.format(league, year))
        matches = json.load(open(os.path.join(MATCHES_DIR, 'matches_{}_{}.json'.format(league, year)), 'r'))
        matches_shots = []
        for idx, match in enumerate(matches):
            if (idx + 1) % 10 == 0:
                print('Process match {}/{}'.format(idx + 1, len(matches)))
            loop.run_until_complete(main(int(match['id'])))

        json.dump(matches_shots, open(os.path.join(MATCHES_SHOTS_DIR, 'matches_shots_{}_{}.json'.format(league, year)), 'w'))
