"""
Download one month of Lichess games, decompress and filter it, in one step.

Module 4 needs a PGN of decent games. Lichess publishes every rated game as
monthly .pgn.zst files at https://database.lichess.org/ (CC0, free to use).
Early months are small: 2013-01 is 17 MB compressed and ~93 MB as PGN.
Months from 2015 on are many GB, so this script refuses them unless you pass
--allow-large.

Filtering matters more than it looks. In 2013 very few players were rated
2000+, so the old suggestion "--min-elo 2000 with a 10-minute minimum" keeps
3 games out of 121,332. The defaults here (both players 1800+, games of at
least 3 minutes) keep 4,794 games from 2013-01, about 320K positions.

Usage:
    python3 training/get_lichess.py                          # 2013-01 -> data/filtered.pgn
    python3 training/get_lichess.py --month 2013-02 --output data/filtered_2013-02.pgn
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.filter_games import filter_games

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--month', default='2013-01', help='YYYY-MM')
    parser.add_argument('--output', default='data/filtered.pgn')
    parser.add_argument('--min-elo', type=int, default=1800, help='both players at least this')
    parser.add_argument('--min-time', type=int, default=180, help='base time in seconds')
    parser.add_argument('--download-dir', default='data/lichess')
    parser.add_argument('--allow-large', action='store_true', help='allow months from 2015 on')
    a = parser.parse_args()

    if a.month >= '2015-01' and not a.allow_large:
        sys.exit(f"{a.month} is several GB compressed. Use a 2013-2014 month, or pass --allow-large.")

    try:
        import zstandard
    except ImportError:
        sys.exit('zstandard is missing: pip install -e ".[train]" (or pip install zstandard)')

    os.makedirs(a.download_dir, exist_ok=True)
    zst = os.path.join(a.download_dir, f"lichess_{a.month}.pgn.zst")
    raw = zst[:-4]
    if not os.path.exists(zst):
        print(f"Downloading {URL.format(month=a.month)}")
        urllib.request.urlretrieve(URL.format(month=a.month), zst + '.part')
        os.replace(zst + '.part', zst)
    print(f"Compressed: {os.path.getsize(zst) / 1e6:.0f} MB")

    print(f"Decompressing to {raw}")
    with open(zst, 'rb') as src, open(raw, 'wb') as dst:
        zstandard.ZstdDecompressor().copy_stream(src, dst)

    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    filter_games(raw, a.output, min_elo=a.min_elo, min_time=a.min_time)
    os.remove(raw)  # the decompressed month is large and no longer needed
    print(f"\nNext: python3 training/train.py --pgn {a.output} --max-games 1000 --epochs 2")


if __name__ == '__main__':
    main()
