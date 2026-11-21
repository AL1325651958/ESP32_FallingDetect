"""Capture a bounded serial run and summarize inference/capture timing."""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import serial


def summarize(text):
    samples=[]
    for line in text.splitlines():
        if not line.startswith('activity '):continue
        samples.append({k:float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)',line)})
    full=[s for s in samples if s.get('buffered')==24]
    updates=[s for s in full if s.get('updated',s.get('inference_us',0)>5000)]
    def stats(values):
        if not values:return None
        values=sorted(values)
        return dict(count=len(values),min=values[0],median=statistics.median(values),
                    p95=values[min(len(values)-1,int(.95*(len(values)-1)))],max=values[-1])
    capture=[s['capture_ms'] for s in samples if 'capture_ms' in s]
    intervals=[(b-a)&0xffffffff for a,b in zip(capture,capture[1:])]
    return dict(activity_frames=len(samples),full_window_frames=len(full),
                full_inference_us=stats([s['inference_us'] for s in updates]),
                capture_interval_ms=stats(intervals),queue_wait_ms=stats([s['queue_ms'] for s in samples if 'queue_ms' in s]),
                queue_drops=max((s.get('queue_drops',0) for s in samples),default=0),
                queue_overrun_messages=text.count('queue overrun'),
                self_test_pass='LSTM self-test PASS' in text,
                ready_10hz='ready: 8x8 at 10 Hz' in text,
                warning_lines=[s for s in text.splitlines() if any(x in s for x in ['panic','Guru Meditation','assert failed','Watchdog','check data ready failed','read ranging data failed',
                    'VL53L8CX not detected','sensor reset failed','ULD init failed','set 8x8 resolution failed',
                    'set ranging frequency failed','set continuous mode failed','start ranging failed'])])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',default='COM5')
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reset',action='store_true')
    args=parser.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    chunks=[]
    with serial.Serial(args.port,115200,timeout=.2) as port:
        if args.reset:
            port.dtr=False;port.rts=True;time.sleep(.1);port.rts=False
        deadline=time.monotonic()+args.seconds
        while time.monotonic()<deadline:
            part=port.read(max(1,port.in_waiting))
            if part:chunks.append(part)
    text=b''.join(chunks).decode('utf-8',errors='replace')
    args.out.write_text(text,encoding='utf-8')
    report=summarize(text);report.update(port=args.port,seconds=args.seconds,log=str(args.out))
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
