# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Real basketball tracking demo: YOLOX ONNX detection + OpenCV motion/geometry.

The ten opening player boxes and six court landmarks are manually initialized.
Every subsequent box is derived from video detections or short optical-flow
continuation; jersey numbers and personal identities are deliberately not inferred.
"""
from pathlib import Path
import argparse, json, time
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
cv2.setNumThreads(4)

class Detector:
    """YOLOX-S decoding following the Apache-2.0 OpenCV Zoo example."""
    def __init__(self, weights):
        self.net = cv2.dnn.readNet(str(weights))
        grids, strides = [], []
        for stride in (8,16,32):
            yy,xx=np.mgrid[:640//stride,:640//stride]
            g=np.stack([xx,yy],axis=-1).reshape(-1,2)
            grids.append(g);strides.append(np.full((len(g),1),stride))
        self.grid=np.concatenate(grids).astype(np.float32)
        self.stride=np.concatenate(strides).astype(np.float32)

    def run(self, frame):
        H,W=frame.shape[:2]
        allboxes,allscores=[],[]
        for x0,x1 in [(0,W),(0,int(W*.6)),(int(W*.4),W)]:
            crop=frame[:,x0:x1];h,w=crop.shape[:2]
            ratio=min(640/h,640/w)
            padded=np.full((640,640,3),114,np.uint8)
            nw,nh=round(w*ratio),round(h*ratio)
            padded[:nh,:nw]=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(nw,nh))
            self.net.setInput(np.transpose(padded.astype(np.float32),(2,0,1))[None])
            out=self.net.forward()[0]
            scores=out[:,4]*out[:,5] # COCO class 0: person
            take=scores>.24
            d=out[take];sc=scores[take]
            center=(d[:,:2]+self.grid[take])*self.stride[take]
            wh=np.exp(d[:,2:4])*self.stride[take]
            xy=(center-wh/2)/ratio;wh/=ratio;xy[:,0]+=x0
            xywh=np.concatenate([xy,wh],axis=1)
            allboxes.extend(xywh.tolist());allscores.extend(sc.tolist())
        inds=cv2.dnn.NMSBoxes(allboxes,allscores,.24,.42)
        result=[]
        for i in np.array(inds).ravel():
            x,y,w,h=allboxes[i]
            box=np.array([max(0,x),max(0,y),min(W-1,x+w),min(H-1,y+h)],float)
            if h<85 or w<24 or w/h>1.25: continue
            result.append({'bbox':box,'confidence':float(allscores[i])})
        return result

def transform_points(points,H):
    return cv2.perspectiveTransform(np.asarray(points,np.float32).reshape(-1,1,2),H).reshape(-1,2)

def iou(a,b):
    x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-6)

def appearance(frame,box):
    x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
    patch=frame[int(y1+.18*h):int(y1+.66*h),int(x1+.20*w):int(x2-.20*w)]
    if patch.size==0:return (.0,.0)
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    blue=np.mean((hsv[:,:,0]>96)&(hsv[:,:,0]<132)&(hsv[:,:,1]>75)&(hsv[:,:,2]>45))
    white=np.mean((hsv[:,:,1]<65)&(hsv[:,:,2]>175))
    return (float(blue),float(white))

def flow_prediction(prevgray,gray,box):
    # Sparse features are taken from each player's inner body, with a
    # forward/backward check to reject unreliable optical flow.
    scale=.5;b=box*scale
    mask=np.zeros(prevgray.shape,np.uint8)
    x1,y1,x2,y2=b;w=x2-x1;h=y2-y1
    cv2.rectangle(mask,(max(0,int(x1+.15*w)),max(0,int(y1+.12*h))),
                  (min(mask.shape[1]-1,int(x2-.15*w)),min(mask.shape[0]-1,int(y2-.12*h))),255,-1)
    pts=cv2.goodFeaturesToTrack(prevgray,45,.015,4,mask=mask)
    if pts is None or len(pts)<4:return box.copy(),False
    new,st,_=cv2.calcOpticalFlowPyrLK(prevgray,gray,pts,None,winSize=(21,21),maxLevel=3)
    if new is None or st is None:return box.copy(),False
    old,back,_=cv2.calcOpticalFlowPyrLK(gray,prevgray,new,None,winSize=(21,21),maxLevel=3)
    if old is None or back is None:return box.copy(),False
    good=(st.ravel()==1)&(back.ravel()==1)&(np.linalg.norm(old-pts,axis=2).ravel()<1.8)
    if good.sum()<4:return box.copy(),False
    shifts=(new[good]-pts[good]).reshape(-1,2)/scale
    shift=np.median(shifts,axis=0)
    if np.linalg.norm(shift)>90:return box.copy(),False
    return box+np.tile(shift,2),True

class CourtMapper:
    def __init__(self,frame,players):
        # Six visually chosen NBA court line intersections in the first frame.
        image=np.float32([[1208,269],[1776,791],[1368,422],[1547,592],[798,468],[909,650]])
        world=np.float32([[28.6512,0],[28.6512,15.24],[28.6512,5.1816],
                          [28.6512,10.0584],[22.86,5.1816],[22.86,10.0584]])
        self.initial,_=cv2.findHomography(image,world,0)
        self.sift=cv2.SIFT_create(nfeatures=2800,contrastThreshold=.02)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.ground=np.int32([[0,351],[1208,269],[1776,791],[0,1040]])
        self.refgray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540))
        mask=np.zeros((540,960),np.uint8);cv2.fillPoly(mask,[self.ground//2],255)
        for b in players:
            x1,y1,x2,y2=(b*.5).astype(int);cv2.rectangle(mask,(x1-8,y1-5),(x2+8,y2+3),0,-1)
        cv2.rectangle(mask,(690,420),(959,539),0,-1)
        self.kp,self.des=self.sift.detectAndCompute(self.refgray,mask)
        self.camera=np.eye(3);self.last_good=0

    def update(self,gray,detections,frame_index):
        mask=np.zeros(gray.shape,np.uint8)
        # Project reference court into the current image using the preceding
        # camera transform, then exclude moving player detections.
        corners=transform_points(self.ground,np.linalg.inv(self.camera))*.5
        cv2.fillPoly(mask,[np.int32(corners)],255)
        for d in detections:
            x1,y1,x2,y2=(d['bbox']*.5).astype(int)
            cv2.rectangle(mask,(x1-6,y1-4),(x2+6,y2+2),0,-1)
        cv2.rectangle(mask,(690,420),(959,539),0,-1)
        kp,des=self.sift.detectAndCompute(gray,mask)
        inliers=0;matches=[]
        if des is not None and self.des is not None:
            pairs=self.matcher.knnMatch(des,self.des,k=2)
            matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.70*pair[1].distance]
        if len(matches)>=12:
            src=np.float32([kp[m.queryIdx].pt for m in matches])*2
            dst=np.float32([self.kp[m.trainIdx].pt for m in matches])*2
            M,ok=cv2.findHomography(src,dst,cv2.RANSAC,4.0,maxIters=2500)
            if M is not None:
                inliers=int(ok.sum());ratio=inliers/len(matches)
                if inliers>=12 and ratio>.35 and np.isfinite(M).all() and .2<abs(np.linalg.det(M[:2,:2]))<5:
                    self.camera=M/M[2,2];self.last_good=frame_index
        return self.initial@self.camera,{'matches':len(matches),'inliers':inliers,'age':frame_index-self.last_good}

def main():
    ap=argparse.ArgumentParser(description='Process the supplied basketball sample with its reviewed calibration. Recalibrate before using a different clip.')
    ap.add_argument('--video',default=str(ROOT/'assets/source.mp4'))
    ap.add_argument('--weights',default=str(ROOT/'assets/yolox_s.onnx'))
    ap.add_argument('--output',default=str(ROOT/'output/tracks.json'))
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    if args.limit<0:ap.error('--limit must be zero (all frames) or a positive number')
    cap=cv2.VideoCapture(args.video);fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,first=cap.read()
    if not ok:raise RuntimeError('Video cannot be read')
    if first.shape[:2]!=(1080,1920):
        cap.release()
        raise ValueError('The bundled calibration expects 1920x1080 footage. Update CourtMapper and the seed boxes for other inputs.')
    if not np.isfinite(fps) or fps<=0 or total<=0:
        cap.release()
        raise ValueError('Video FPS and frame count must be valid and positive')
    # Manual initialization fixes the intended ten players, excluding officials
    # and spectators. Detection, matching, and motion thereafter are automatic.
    seeds=[(0,[294,535,467,764]),(0,[771,407,872,626]),(0,[865,225,924,412]),
           (0,[1233,387,1338,557]),(0,[1297,508,1379,741]),
           (1,[263,464,396,704]),(1,[497,517,619,747]),(1,[666,154,789,331]),
           (1,[1339,626,1438,859]),(1,[1374,463,1512,701])]
    tracks=[{'id':i+1,'team':team,'bbox':np.array(box,float),'miss':0,'court':None}
            for i,(team,box) in enumerate(seeds)]
    mapper=CourtMapper(first,[t['bbox'] for t in tracks]);detector=Detector(args.weights)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0);frames=[];cache=[];prevgray=None;start=time.time()
    for n in range(total if not args.limit else min(total,args.limit)):
        ok,frame=cap.read()
        if not ok:break
        gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540))
        detections=detector.run(frame)
        H,cal=mapper.update(gray,detections,n) if n else (mapper.initial,{'matches':0,'inliers':0,'age':0})
        candidates=[]
        for d in detections:
            b=d['bbox'];foot=[(b[0]+b[2])/2,b[3]];p=transform_points([foot],H)[0]
            if not (8<p[0]<28.90 and -.35<p[1]<15.65):continue
            d['court']=p;d['appearance']=appearance(frame,b)
            candidates.append(d)
        cache.append({'frame':n,'homography':H.tolist(),'camera':mapper.camera.tolist(),
                      'candidates':[{'bbox':d['bbox'].tolist(),'court':d['court'].tolist(),
                                     'confidence':d['confidence'],'appearance':d['appearance']}
                                    for d in candidates]})
        predicted=[]
        for t in tracks:
            p,valid=flow_prediction(prevgray,gray,t['bbox']) if prevgray is not None else (t['bbox'],True)
            predicted.append((p,valid))
        cost=np.full((len(tracks),len(candidates)),100.,float)
        for i,t in enumerate(tracks):
            p=predicted[i][0];center=(p[:2]+p[2:])/2;ph=max(80,p[3]-p[1])
            for j,d in enumerate(candidates):
                b=d['bbox'];dist=np.linalg.norm((b[:2]+b[2:])/2-center)/ph
                overlap=iou(p,b);blue,white=d['appearance']
                # Only strong color evidence penalizes a cross-team match.
                mismatch=(t['team']==0 and blue>.18 and blue>white*1.4) or (t['team']==1 and white>.33 and blue<.06)
                size=abs(np.log(max(30,b[3]-b[1])/ph))
                cost[i,j]=.9*(1-overlap)+1.2*dist+.35*size+(1.5 if mismatch else 0)-.10*d['confidence']
                if dist>1.0 or size>.9:cost[i,j]=100
                # A blue-uniform track must never acquire a gray official.
                if t['team']==1 and blue<.07:cost[i,j]=100
                if t['team']==0 and (white<.075 or blue>.22):cost[i,j]=100
        assigned={}
        if len(candidates):
            ii,jj=linear_sum_assignment(cost)
            assigned={int(i):int(j) for i,j in zip(ii,jj) if cost[i,j]<1.45}
        # Recover briefly occluded players from remaining detections using
        # ground-plane proximity and positive uniform evidence. Established
        # matches are preserved so a recovering track cannot steal a player.
        lost=[i for i,t in enumerate(tracks) if i not in assigned and t['miss']>=3]
        remaining=[j for j in range(len(candidates)) if j not in assigned.values()]
        if lost and remaining:
            recovery=np.full((len(lost),len(remaining)),100.,float)
            for a,i in enumerate(lost):
                t=tracks[i]
                if t['court'] is None:continue
                for z,j in enumerate(remaining):
                    d=candidates[j];blue,white=d['appearance']
                    valid=(blue>.10) if t['team']==1 else (white>.10 and blue<.16)
                    gap=float(np.linalg.norm(d['court']-t['court']))
                    h0=max(80,t['bbox'][3]-t['bbox'][1]);h1=max(80,d['bbox'][3]-d['bbox'][1])
                    size=abs(np.log(h1/h0))
                    if valid and gap<min(6.,1.5+t['miss']*.12) and size<.7:
                        recovery[a,z]=gap+.8*size-.2*d['confidence']
            aa,zz=linear_sum_assignment(recovery)
            for a,z in zip(aa,zz):
                if recovery[a,z]<6:assigned[lost[a]]=remaining[z]
        visible=[]
        for i,t in enumerate(tracks):
            observed=i in assigned
            if observed:
                d=candidates[assigned[i]];t['bbox']=d['bbox'];t['miss']=0;confidence=d['confidence']
            else:
                t['bbox']=predicted[i][0];t['miss']+=1;confidence=max(0,.45-.04*t['miss'])
            if t['miss']>7:continue
            b=t['bbox'];raw=transform_points([[(b[0]+b[2])/2,b[3]]],H)[0]
            # Light temporal smoothing reduces box-foot jitter in the mini-map.
            court=raw if t['court'] is None else .55*raw+.45*t['court']
            t['court']=court
            if not (-.6<court[1]<15.8 and 7<court[0]<28.95):continue
            visible.append({'id':t['id'],'team':t['team'],'bbox':[round(float(x),2) for x in b],
                            'court':[round(float(x),4) for x in court],
                            'confidence':round(confidence,4),'observed':observed,
                            'continuation_frames':t['miss']})
        frames.append({'frame':n,'time':round(n/fps,5),'players':visible,'calibration':cal})
        prevgray=gray
        if n%30==0 or n==total-1:
            print(json.dumps({'frame':n,'total':total,'visible':len(visible),'detected':sum(p['observed'] for p in visible),
                              'court_inliers':cal['inliers'],'elapsed_s':round(time.time()-start,1)}),flush=True)
    cap.release()
    data={'fps':fps,'width':first.shape[1],'height':first.shape[0],'frames':frames,
          'meta':{'calibration':'Six manually initialized court landmarks; per-frame SIFT/RANSAC camera compensation; estimated ground-plane positions.',
                  'source_label':'Roboflow basketball tutorial sample',
                  'source_url':'https://roboflow.com/blog/identify-basketball-players',
                  'initialization':'Ten manually selected first-frame player boxes; automatic YOLOX detection and optical-flow-assisted association thereafter.',
                  'detector':'OpenCV Zoo YOLOX-S ONNX, COCO person class; three overlapping image views per frame',
                  'teams':{'0':'White uniforms','1':'Blue uniforms'},'identities':'Tracking IDs only; no name or jersey recognition.',
                  'limitations':'Experimental short-clip demo; occlusion can cause ID switches; court positions approximate; no verified speeds or distances.'}}
    out=Path(args.output);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(data))
    (out.parent/'tracks_automatic.json').write_text(json.dumps(data))
    (out.parent/'detection_cache.json').write_text(json.dumps(cache))
    counts=Counter(len(f['players']) for f in frames)
    obs=sum(p['observed'] for f in frames for p in f['players']);allp=sum(len(f['players']) for f in frames)
    summary={'frames':len(frames),'duration':len(frames)/fps,'player_counts':dict(counts),'detected_observations':obs,
             'short_flow_continuations':allp-obs,'calibration_stale_frames':sum(f['calibration']['age']>0 for f in frames),
             'processing_seconds':round(time.time()-start,2)}
    (out.parent/'processing_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
