"""Synchronised video + ACTUAL Thermopile.processing stages for a recorded session.

Run from the project root:
    python -m Tools.thermal_stage_playback --recording "Datasets/training/.../recording_..."

Requirements: pip install numpy pandas scipy opencv-python

Keys:
    SPACE play/pause     A/D or left/right -/+1 sec   J/L -/+5 sec
    1..9 stage selection  0 next stage                 V next, C previous
    [ / ] frame step (paused)       R restart          Q/Esc quit

Use the seek bar to inspect any moment. This script NEVER modifies the recording.
It uses saved metadata.json background and sigma, then sequentially processes all
thermal frames via the *real* Thermopile.processing.ThermopilePipeline class.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import package from project root even when this script is run by filepath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Thermopile.processing import ThermopilePipeline

STAGES = [
    ('Raw temperature', 'raw', 'C'),
    ('Estimated background (before update)', 'background', 'C'),
    ('Signed difference (frame - bg)', 'signed', 'C'),
    ('Absolute temperature difference', 'absolute', 'C'),
    ('Background noise sigma (before update)', 'sigma', 'C'),
    ('Raw absolute Z score', 'z', 'sigma'),
    ('Gaussian-filtered Z score', 'filtered', 'sigma'),
    ('Threshold mask (detections)', 'mask', 'binary'),
    ('Foreground mask over raw temperature', 'mask_overlay', 'overlay'),
    ('Detected blobs', 'blobs', 'overlay'),
    ('Blob tracks and primary track', 'tracks', 'overlay'),
]
WINDOW = 'Thermopile stages | synchronised recording'
WIDTH, HEIGHT = 1500, 825
BAR_Y = HEIGHT - 75
TRACKBAR_MAX = 10000
NAVIGATING = False


def load_recording(folder: Path):
    video = next((p for p in (folder/'video.mp4', folder/'video_labelstudio.mp4') if p.is_file()), None)
    if video is None:
        raise FileNotFoundError(f'No video.mp4/video_labelstudio.mp4 in {folder}')
    for name in ('thermopile.csv', 'video_frames.csv', 'metadata.json'):
        if not (folder/name).is_file():
            raise FileNotFoundError(f'Missing {folder/name}')
    thermal_df = pd.read_csv(folder/'thermopile.csv')
    video_df = pd.read_csv(folder/'video_frames.csv')
    with (folder/'metadata.json').open(encoding='utf-8') as f:
        meta = json.load(f)
    shape = tuple(int(v) for v in meta.get('thermal_frame_shape', (24,32)))
    if shape != (24,32):
        raise ValueError(f'Expected a 24x32 recording, got {shape}')
    cols = [f'pixel_{i}' for i in range(24*32)]
    if not set(cols).issubset(thermal_df.columns):
        raise ValueError('thermopile.csv must contain pixel_0 through pixel_767')
    if meta.get('thermal_background') is None or meta.get('thermal_sigma') is None:
        raise ValueError('metadata.json must contain the saved thermal_background and thermal_sigma. '
                         'Do not infer initial calibration from footage containing a person.')
    frames = thermal_df[cols].to_numpy(dtype=np.float64).reshape(-1,24,32)
    background = np.asarray(meta['thermal_background'], dtype=np.float64).reshape(24,32)
    sigma = np.asarray(meta['thermal_sigma'], dtype=np.float64).reshape(24,32)
    if not np.isfinite(frames).all():
        raise ValueError('Thermal recording has non-finite pixel values; inspect source data before replay')
    if not np.isfinite(background).all() or not np.isfinite(sigma).all():
        raise ValueError('Background calibration contains invalid values')
    if (sigma <= 0).any():
        raise ValueError('Saved background sigma contains zero/negative values')
    def time_col(df):
        return next((c for c in ('time_s','pc_timestamp','rock_timestamp') if c in df.columns), None)
    vc, tc = time_col(video_df), time_col(thermal_df)
    if vc is None or tc is None:
        raise ValueError('CSV files require time_s (or compatible timestamp columns)')
    vt = video_df[vc].to_numpy(dtype=float)
    tt = thermal_df[tc].to_numpy(dtype=float)
    # Match existing synchronised_playback: relative time_s already has same origin.
    if not (vc == tc == 'time_s'):
        origin = vt[0]
        vt, tt = vt-origin, tt-origin
    if not np.isfinite(vt).all() or not np.isfinite(tt).all():
        raise ValueError('Recording timestamps include invalid values')
    if np.any(np.diff(vt) < 0) or np.any(np.diff(tt) < 0):
        raise ValueError('Timestamps must be in acquisition order')
    if not len(vt) or not len(tt):
        raise ValueError('Recording has no frames')
    return video, frames, vt, tt, background, sigma


def process_all(frames, initial_background, initial_sigma):
    """Replay in the original time order: safe, deterministic scrubbing later."""
    pipe = ThermopilePipeline((24,32))
    pipe.bg_model.bg[:] = initial_background
    pipe.bg_model.sigma[:] = initial_sigma
    pipe.bg_model._initialised = True
    collected = {k: [] for _,k,_ in STAGES if k not in ('blobs','tracks','mask_overlay')}
    annotations = []
    for idx, frame in enumerate(frames):
        # BEFORE process_frame: updated bg/sigma are for the NEXT frame.
        bg_before = pipe.bg_model.bg.copy()
        sig_before = pipe.bg_model.sigma.copy()
        signed = frame - bg_before
        detections = pipe.process_frame(frame)
        z_filtered = np.asarray(pipe.last_smoothed_z)
        # Latest processing.py uses the SAME threshold mask for detection
        # and for freezing foreground pixels in the background update.
        data = {
            'raw': frame,
            'background': bg_before,
            'signed': signed,
            'absolute': np.abs(signed),
            'sigma': sig_before,
            'z': pipe.last_raw_z,
            'filtered': z_filtered,
            'mask': pipe._prev_mask,
        }
        for key, array in data.items():
            collected[key].append(np.asarray(array, dtype=np.float32).copy())
        entries = []
        for item in detections:
            b = item.blob
            f = item.features
            entries.append((b.bbox, b.centroid, f.track_id, b.pixel_count,
                            f.track_age_frames, f.track_id == pipe.primary_track_id))
        annotations.append(entries)
        if (idx+1) % 500 == 0:
            print(f'  Replayed {idx+1}/{len(frames)} thermal frames', flush=True)
    arrays = {k: np.stack(v) for k,v in collected.items()}
    return arrays, annotations


def nearest_index(times, t):
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times)-1
    return i if abs(times[i]-t) < abs(times[i-1]-t) else i-1


def image_range(values, min_width=1.0):
    lo, hi = np.percentile(values, (2, 98))
    center = 0.5*(lo+hi)
    width = max(float(hi-lo), min_width)
    return center-width/2, center+width/2


def render_stage(arrays, annotations, idx, mode, raw_range):
    label, key, units = STAGES[mode]
    raw = arrays['raw'][idx]
    a = arrays[key][idx] if key in arrays else arrays['raw'][idx]
    if key in ('blobs','tracks','mask_overlay'):
        lo, hi = raw_range
        norm = np.uint8(np.clip((raw-lo)/max(hi-lo, 1e-8)*255,0,255))
        out = cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
        # Draw on native pixels after enlargement for clearly legible boxes.
        large = cv2.resize(out,(704,528),interpolation=cv2.INTER_NEAREST)
        sx, sy = 704/32, 528/24
        if key == 'mask_overlay':
            mask_big = cv2.resize((arrays['mask'][idx] > 0).astype(np.uint8),
                                  (704,528), interpolation=cv2.INTER_NEAREST).astype(bool)
            tint = np.zeros_like(large)
            tint[:,:,1] = 255
            large[mask_big] = cv2.addWeighted(large, 0.50, tint, 0.50, 0)[mask_big]
            return large, f"{int(arrays['mask'][idx].sum())} foreground pixels (green); same mask protects background"
        for bbox, centroid, track_id, count, age, primary in annotations[idx]:
            r0,c0,r1,c1 = bbox
            clr = (0,245,255) if primary else (70,235,85)
            cv2.rectangle(large,(int(c0*sx),int(r0*sy)),
                          (int((c1+1)*sx)-1,int((r1+1)*sy)-1),clr,2)
            cx,cy = int((centroid[1]+.5)*sx),int((centroid[0]+.5)*sy)
            cv2.circle(large,(cx,cy),4,clr,-1)
            if key == 'tracks':
                cv2.putText(large,f'ID {track_id}  age {age}',(max(0,cx-15), max(17,cy-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,clr,1,cv2.LINE_AA)
        return large, f'{len(annotations[idx])} blobs  |  primary: yellow'
    if units == 'binary':
        norm = (a>0).astype(np.uint8)*255
        im = cv2.applyColorMap(norm,cv2.COLORMAP_BONE)
        return cv2.resize(im,(704,528),interpolation=cv2.INTER_NEAREST), f'{int((a>0).sum())} active pixels'
    if key == 'raw' or key == 'background':
        lo,hi = raw_range
        norm = np.uint8(np.clip((a-lo)/max(hi-lo, 1e-8)*255,0,255))
        im = cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
    elif key == 'signed':
        m = max(0.5,float(np.percentile(np.abs(arrays['signed']),99)))
        norm = np.uint8(np.clip((a+m)/(2*m)*255,0,255))
        im = cv2.applyColorMap(norm,cv2.COLORMAP_COOLWARM) if hasattr(cv2,'COLORMAP_COOLWARM') else cv2.applyColorMap(norm,cv2.COLORMAP_TURBO)
        return cv2.resize(im,(704,528),interpolation=cv2.INTER_NEAREST), f'Blue/cool  -{m:.2f} C   |   red/hot  +{m:.2f} C'
    else:
        # Fixed contrast across the WHOLE RECORDING for meaningful comparisons.
        positive = arrays[key]
        high = max(0.2,float(np.percentile(positive,99.5)))
        norm = np.uint8(np.clip(a/high*255,0,255))
        im = cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
        return cv2.resize(im,(704,528),interpolation=cv2.INTER_NEAREST), f'0 .. {high:.2f} {units}'
    return cv2.resize(im,(704,528),interpolation=cv2.INTER_NEAREST), f'{float(np.min(a)):.2f} .. {float(np.max(a)):.2f} C'


def fit_image(img, w, h):
    ih,iw=img.shape[:2]
    factor=min(w/iw,h/ih)
    out=cv2.resize(img,(max(1,int(iw*factor)),max(1,int(ih*factor))))
    canvas=np.full((h,w,3),(31,33,39),np.uint8)
    x=(w-out.shape[1])//2;y=(h-out.shape[0])//2
    canvas[y:y+out.shape[0],x:x+out.shape[1]]=out
    return canvas


def text(img,s,x,y,size=.65,color=(235,235,235),thick=1):
    cv2.putText(img,str(s),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thick,cv2.LINE_AA)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--stage',type=int,default=1,help='Starting stage number 1..11 (default 1)')
    args=parser.parse_args()
    folder=args.recording.resolve()
    video, raw, vt, tt, bg, sigma=load_recording(folder)
    print('Replaying recorded thermopile frames through your actual pipeline...',flush=True)
    arrays, annotations=process_all(raw,bg,sigma)
    raw_range=image_range(arrays['raw'],min_width=4)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV cannot open video: {video}')
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(len(vt), reported_count) if reported_count > 0 else len(vt)
    if count<=0:
        raise RuntimeError('No video frames')
    vt=vt[:count]
    stage=max(0,min(len(STAGES)-1,args.stage-1))
    state={'frame':0,'paused':True,'seek':None}
    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW,WIDTH,HEIGHT)
    def scrub(value):
        if not NAVIGATING:
            state['seek']=int(round(value/TRACKBAR_MAX*(count-1)))
            state['paused']=True
    cv2.createTrackbar('Timeline',WINDOW,0,TRACKBAR_MAX,scrub)
    start_wall=time.perf_counter()
    start_data=float(vt[0])
    last_frame=-1
    video_img=None
    try:
        while True:
            if state['seek'] is not None:
                state['frame']=state['seek']
                state['seek']=None
            vi=state['frame']
            if vi!=last_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES,vi)
                ok,video_img=cap.read()
                if not ok:
                    state['paused']=True
                    if last_frame>=0:
                        state['frame']=last_frame
                    continue
                last_frame=vi
            stamp=float(vt[vi]); ti=nearest_index(tt,stamp)
            screen=np.full((HEIGHT,WIDTH,3),(23,25,30),np.uint8)
            text(screen,'RECORDED VIDEO',25,40,.83,(235,240,245),2)
            text(screen,f'THERMOPILE: {STAGES[stage][0]}',770,40,.72,(220,233,252),2)
            screen[65:595,25:739]=fit_image(video_img,714,530)
            thermal_view,desc=render_stage(arrays,annotations,ti,stage,raw_range)
            screen[65:593,772:1476]=thermal_view
            text(screen,f'Video t={stamp:.3f}s  frame={vi+1}/{count}',28,633,.58)
            text(screen,f'Thermal t={tt[ti]:.3f}s   dt={tt[ti]-stamp:+.3f}s  index={ti+1}/{len(tt)}',772,633,.55)
            text(screen,desc,772,662,.55,(140,220,188))
            text(screen,'SPACE Play/Pause  |  A/D +/-1s  J/L +/-5s  [/] single frame  |  R restart',25,707,.53)
            text(screen,'1..9 select stage  |  0 or V next  C previous  |  Q quit',25,733,.54)
            text(screen,f'Stage {stage+1}/{len(STAGES)} - {STAGES[stage][0]}',25,764,.58,(140,220,188))
            text(screen,'PAUSED' if state['paused'] else 'PLAYING',1300,763,.60,(90,195,255))
            global NAVIGATING
            NAVIGATING=True
            cv2.setTrackbarPos('Timeline',WINDOW,int(round(vi/max(1,count-1)*TRACKBAR_MAX)))
            NAVIGATING=False
            cv2.imshow(WINDOW,screen)
            key=cv2.waitKeyEx(20)
            if key in (27,ord('q')):
                break
            if key==ord(' '):
                state['paused']=not state['paused']
                if not state['paused']:
                    start_wall=time.perf_counter();start_data=stamp
            elif key in (ord('a'),2424832,81):
                state['frame']=nearest_index(vt,max(float(vt[0]),stamp-1));state['paused']=True
            elif key in (ord('d'),2555904,83):
                state['frame']=nearest_index(vt,min(float(vt[-1]),stamp+1));state['paused']=True
            elif key==ord('j'):
                state['frame']=nearest_index(vt,max(float(vt[0]),stamp-5));state['paused']=True
            elif key==ord('l'):
                state['frame']=nearest_index(vt,min(float(vt[-1]),stamp+5));state['paused']=True
            elif key==ord('['):
                state['frame']=max(0,vi-1);state['paused']=True
            elif key==ord(']'):
                state['frame']=min(count-1,vi+1);state['paused']=True
            elif key in (ord('v'),ord('0')):
                stage=(stage+1)%len(STAGES)
            elif key==ord('c'):
                stage=(stage-1)%len(STAGES)
            elif ord('1')<=key<=ord('9'):
                stage=key-ord('1')
            elif key==ord('r'):
                state['frame']=0;state['paused']=True
            if not state['paused']:
                target=start_data+(time.perf_counter()-start_wall)
                if target>=vt[-1]:
                    state['frame']=count-1;state['paused']=True
                else:
                    state['frame']=nearest_index(vt,target)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
