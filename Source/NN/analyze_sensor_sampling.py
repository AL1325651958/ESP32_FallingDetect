"""Summarize raw ToF status snapshots independently of neural-network output."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from capture_fall_device import summarize


def stats(values):
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                max=max(values)) if values else None


def analyze(text):
    snapshots, configurations, malformed = [], [], []
    names = ('raw_mm', 'status', 'targets', 'sigma_mm', 'signal_kcps_spad', 'ambient_kcps_spad')
    for line in text.splitlines():
        for prefix, destination in (('sensor_diag ', snapshots), ('sensor_config ', configurations)):
            if not line.startswith(prefix):
                continue
            try:
                item = json.loads(line[len(prefix):])
                if destination is snapshots and any(len(item[name]) != 64 for name in names):
                    raise ValueError('Expected six aligned 64-zone arrays')
                destination.append(item)
            except (ValueError, KeyError, TypeError) as error:
                malformed.append(str(error))
    statuses, rejected, near_rejected = Counter(), Counter(), Counter()
    valid_counts, near_counts, background_counts = [], [], []
    valid_distances, valid_signal, rejected_signal, ambient, sigma = [], [], [], [], []
    for frame in snapshots:
        valid = near = background = 0
        for z in range(64):
            status, distance = frame['status'][z], frame['raw_mm'][z]
            statuses[status] += 1
            ambient.append(frame['ambient_kcps_spad'][z])
            if status in (5, 9) and distance >= 0:
                valid += 1
                near += distance <= 1000
                background += distance > 1000
                valid_distances.append(distance)
                valid_signal.append(frame['signal_kcps_spad'][z])
                sigma.append(frame['sigma_mm'][z])
            else:
                rejected[status] += 1
                rejected_signal.append(frame['signal_kcps_spad'][z])
                if 0 <= distance <= 1000:
                    near_rejected[status] += 1
        valid_counts.append(valid)
        near_counts.append(near)
        background_counts.append(background)
    activity_valid = [int(x) for x in re.findall(r'^activity .*?\bvalid=(\d+)', text, re.MULTILINE)]
    return dict(runtime=summarize(text), activity_valid_zones=stats(activity_valid),
        configurations=configurations, snapshot_count=len(snapshots), malformed=malformed,
        snapshot_valid_zones=stats(valid_counts), snapshot_valid_at_most_1m_zones=stats(near_counts),
        snapshot_valid_beyond_1m_zones=stats(background_counts),
        status_histogram=dict(sorted(statuses.items())), rejected_status_histogram=dict(sorted(rejected.items())),
        rejected_untrusted_raw_values_at_most_1m_by_status=dict(sorted(near_rejected.items())),
        valid_distance_mm=stats(valid_distances), valid_signal_kcps_spad=stats(valid_signal),
        rejected_signal_kcps_spad=stats(rejected_signal), ambient_kcps_spad=stats(ambient),
        valid_sigma_mm=stats(sigma),
        evidence_limits=[
            'Invalid-status raw distances are untrusted, not proof of a detected near target.',
            'Valid background ranges do not prove that a person or clothing was detected.',
            'Scenes and target placement are not controlled; total valid counts cannot establish sensitivity.',
            'Continuous mode does not use the autonomous integration-time register as a fixed exposure setting.',
            'Signal/ambient are integer kcps/SPAD values as converted by the installed ULD.',
            'Diagnostic snapshots are limited to once a second and do not reveal every transient dropout.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    report=analyze(args.log.read_text('utf-8', errors='replace'))
    report['source_log']=str(args.log)
    out=args.out or args.log.with_suffix('.analysis.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
