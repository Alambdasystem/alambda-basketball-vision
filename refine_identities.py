# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Associate real detections using visually reviewed identity anchors.

Anchor interpolation guides which detection belongs to which player; it is
never presented as a measured player position. Missing detections are omitted.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from process_clip import transform_points, iou

ROOT=Path(__file__).resolve().parent

def read_anchors(path):
    obj=json.loads(path.read_text())
    return obj if isinstance(obj,list) else obj.get('frames',obj.get('anchors',[]))

def mapped_box(box,H):
    x1,y1,x2,y2=box
    corners=transform_points([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],H)
    return np.r_[corners.min(axis=0),corners.max(axis=0)]

def main():
    folder=ROOT/'output'
    required=['tracks_automatic.json','detection_cache.json','identity_anchors.json','blue_identity_anchors.json']
    missing=[name for name in required if not (folder/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs: '+', '.join(missing)+'. Run process_clip.py before refinement; preserve the reviewed anchor files.')
    data=json.loads((folder/'tracks_automatic.json').read_text())
    cache=json.loads((folder/'detection_cache.json').read_text())
    anchors=read_anchors(folder/'identity_anchors.json')+read_anchors(folder/'blue_identity_anchors.json')
    if len(data['frames'])!=len(cache):
        raise ValueError('Automatic tracks and detector cache have different frame counts. Rerun process_clip.py.')
    series={i:{} for i in range(1,11)}
    for item in anchors:
        n=int(item['frame'])
        if not 0<=n<len(cache):
            raise ValueError('Reviewed anchors extend beyond this run. Process the full sample before refining.')
        H=np.array(cache[n]['camera'])
        for p in item['players']:
            box=np.array(p['bbox'],float);pid=int(p['id'])
            if pid not in series:continue
            series[pid][n]=mapped_box(box,H)
    interpolators={}
    for pid,records in series.items():
        times=np.array(sorted(records));boxes=np.array([records[n] for n in times])
        if len(times)<2:raise RuntimeError(f'Not enough reviewed anchors for track {pid}')
        interpolators[pid]=(PchipInterpolator(times,boxes,axis=0,extrapolate=True),times[0],times[-1])
    previous={};accepted=0;missing=0;review=[]
    for f,c in zip(data['frames'],cache):
        n=f['frame'];camera=np.array(c['camera']);H=np.array(c['homography']);inv=np.linalg.inv(camera)
        expected={}
        for pid,(fn,lo,hi) in interpolators.items():
            refbox=fn(np.clip(n,lo,hi));b=mapped_box(refbox,inv)
            expected[pid]=b
        ds=c['candidates'];ids=list(range(1,11));cost=np.full((10,len(ds)),100.,float)
        for i,pid in enumerate(ids):
            ex=expected[pid];h=max(80,ex[3]-ex[1]);center=(ex[:2]+ex[2:])/2
            exfoot=transform_points([[(ex[0]+ex[2])/2,ex[3]]],H)[0]
            team=0 if pid<=5 else 1
            for j,d in enumerate(ds):
                box=np.array(d['bbox']);blue,white=d['appearance']
                valid=(blue>=.055) if team else (white>=.065 and blue<.24)
                if not valid:continue
                dist=np.linalg.norm((box[:2]+box[2:])/2-center)/h
                ground=np.linalg.norm(np.array(d['court'])-exfoot)
                size=abs(np.log(max(80,box[3]-box[1])/h))
                if dist>.8 or ground>3.0 or size>.65:continue
                cost[i,j]=1.5*dist+.6*(1-iou(box,ex))+.25*ground+.4*size-.15*d['confidence']
        ii,jj=linear_sum_assignment(cost)
        selected={int(i):int(j) for i,j in zip(ii,jj) if cost[i,j]<1.8}
        players=[]
        for i,pid in enumerate(ids):
            if i not in selected:missing+=1;continue
            d=ds[selected[i]];box=np.array(d['bbox']);raw=np.array(d['court']);team=0 if pid<=5 else 1
            if pid in previous and previous[pid][0]==n-1:
                pos=.6*raw+.4*previous[pid][1]
            else:pos=raw
            previous[pid]=(n,pos)
            players.append({'id':pid,'team':team,'bbox':[round(float(x),2) for x in box],
                            'court':[round(float(x),4) for x in pos],
                            'confidence':round(d['confidence'],4),'observed':True,
                            'continuation_frames':0,'association':'reviewed_identity_anchors'})
            accepted+=1
        f['players']=players
        if n in (0,30,60,90,120,150,180,210,237):
            review.append({'frame':n,'ids':[p['id'] for p in players]})
    data['meta']['initialization']='Ten starting players and additional identity anchors at nine reviewed frames; candidate associations are guided by these anchors.'
    data['meta']['identity_review']='Original player correspondence reviewed using jersey appearance and motion continuity; no facial identity or automatic name recognition.'
    data['meta']['measurements']='Every displayed box and court point comes from a real YOLOX detection; interpolated anchor positions only guide association and are never rendered as observations.'
    data['meta']['limitations']='Assisted short-clip demonstration; detections can disappear under occlusion and identities are not statistically validated. Court positions are approximate.'
    (folder/'tracks.json').write_text(json.dumps(data))
    summary={'frames':len(data['frames']),'duration':len(data['frames'])/data['fps'],
             'reviewed_anchors':sum(len(x) for x in series.values()),'detected_observations':accepted,
             'undetected_player_frames':missing,'short_flow_continuations':0,
             'per_frame_min':min(len(f['players']) for f in data['frames']),
             'per_frame_max':max(len(f['players']) for f in data['frames']),
             'keyframe_review':review,
             'note':'Counts are processing statistics, not independently measured accuracy.'}
    (folder/'processing_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
