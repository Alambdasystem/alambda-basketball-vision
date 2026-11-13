# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Download the two public inputs used by this specific demonstration."""
from pathlib import Path
import argparse, hashlib, urllib.request

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
FILES=[
    ('source.mp4','https://drive.google.com/uc?export=download&id=1It3wz3eoGcjo6tI9OGUt3X69ZK6x5SNK',
     '690d5f94e1266e2c3b0935c2e76eee3c2e9831737cdcdea15930e58a7474500d'),
    ('yolox_s.onnx','https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx',
     'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'),
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only',action='store_true',help='Download only the clip, for rendering the included tracks.')
    args=parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    for name,url,expected in FILES:
        if args.source_only and name!='source.mp4':continue
        destination=ASSETS/name
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:
            print('Downloading',name,flush=True)
            temporary=destination.with_suffix(destination.suffix+'.part')
            try:
                with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as output:
                    while chunk:=response.read(1024*1024):output.write(chunk)
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
                    raise RuntimeError(f'Checksum mismatch: {name}')
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        digest=hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest!=expected:raise RuntimeError(f'Checksum mismatch: {name}')
        print('Verified',name)

if __name__=='__main__':main()
