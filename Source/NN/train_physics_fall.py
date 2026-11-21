"""Train a local space-time differential, physics-guided 3-state LSTM."""
from __future__ import annotations
import csv, hashlib, io, json, random, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'NN/artifacts/physics_fall'
WINDOW,PROJECT,LOCAL,HIDDEN,DENSE=24,16,16,24,16
GAP_MIN,GAP_MAX,MIN_VALID,SCALE=40,175,4,4000.0
FEATURE_VERSION=2
AX=np.tile(np.linspace(-1,1,8,dtype=np.float32),8)
AY=np.repeat(np.linspace(-1,1,8,dtype=np.float32),8)
LABEL={'运动':0,'跌倒':1,'有人静止':2}
REVIEW_PATH=ROOT/'NN/review/annotations.json'


def load_review_annotations(path=REVIEW_PATH):
    if not path.exists():
        return {}
    document=json.loads(path.read_text('utf-8'))
    return document.get('records', {})


def reviewed_for(source, annotations):
    """Match sidecar records by source path; IDs are intentionally not shared."""
    source_key=source.replace('/', '\\').casefold()
    for item in annotations.values():
        if str(item.get('source','')).replace('/', '\\').casefold()==source_key:
            return item
    return None


def label_spans_for(frames, original_label, review, digest):
    """Resolve the same whole-label/range precedence as the review exporter.

    Adjacent equal labels merge, so annotating a subset with the same class
    does not unnecessarily sever a continuous 24-frame training sequence.
    """
    if review is None:
        return [(0, frames, original_label)]
    if review.get('sha256') != digest:
        raise ValueError('Review source has changed; review it again before training')
    if review.get('excluded'):
        return []
    whole=review.get('label')
    if whole is not None and whole not in (*LABEL, '无人静止'):
        raise ValueError('Unknown reviewed whole-record label')
    assigned=np.full(frames,LABEL.get(whole,-1),dtype=np.int8)
    occupied=np.zeros(frames,bool)
    for item in review.get('ranges',[]):
        start,end=item.get('start'),item.get('end')
        if (type(start) is not int or type(end) is not int or not 0<=start<=end<frames
                or item.get('label') not in (*LABEL,'无人静止') or occupied[start:end+1].any()):
            raise ValueError('Invalid or overlapping reviewed ranges')
        occupied[start:end+1]=True
        assigned[start:end+1]=LABEL.get(item['label'],-1)
    boundaries=np.r_[0,np.flatnonzero(np.diff(assigned))+1,frames]
    return [(int(a),int(b),int(assigned[a])) for a,b in zip(boundaries[:-1],boundaries[1:]) if assigned[a]>=0]


def assign_splits(records):
    still=[r for r in records if any(label==2 for _,_,label in r['label_spans'])]
    if still:
        held=max(still,key=lambda r:max((b-a for a,b in r['segments']),default=0))
        # A mixed-label source remains in the same split in its entirety.
        held['split']='validation'
    # Byte-identical sources belong to one split even if copied to another ZIP.
    validation_hashes={r['sha256'] for r in records if r['split']=='validation'}
    for r in records:
        if r['sha256'] in validation_hashes:r['split']='validation'
    return records

def features(raw):
    valid=(raw>=0)&(raw!=65535)
    d=np.where(valid,np.clip(raw,0,SCALE)/SCALE,0).astype(np.float32)
    grid=d.reshape(*d.shape[:-1],8,8); mask=valid.reshape(*valid.shape[:-1],8,8)
    # A derivative exists only when its center and every stencil neighbor are
    # measured. Zero-filled missing zones must never create artificial edges.
    gx=np.zeros_like(grid); gy=np.zeros_like(grid)
    gx[...,1:-1]=(grid[...,2:]-grid[...,:-2])*.5*(mask[...,1:-1]&mask[...,2:]&mask[...,:-2])
    gx[...,0]=(grid[...,1]-grid[...,0])*(mask[...,1]&mask[...,0])
    gx[...,-1]=(grid[...,-1]-grid[...,-2])*(mask[...,-1]&mask[...,-2])
    gy[...,1:-1,:]=(grid[...,2:,:]-grid[...,:-2,:])*.5*(mask[...,1:-1,:]&mask[...,2:,:]&mask[...,:-2,:])
    gy[...,0,:]=(grid[...,1,:]-grid[...,0,:])*(mask[...,1,:]&mask[...,0,:])
    gy[...,-1,:]=(grid[...,-1,:]-grid[...,-2,:])*(mask[...,-1,:]&mask[...,-2,:])
    pg=np.concatenate((grid[...,:1,:,:],grid[...,:-1,:,:]),axis=-3)
    pm=np.concatenate((mask[...,:1,:,:],mask[...,:-1,:,:]),axis=-3)
    dt=(grid-pg)* (mask&pm)
    weight=np.where(valid,(1-d)**2,0); mass=np.maximum(weight.sum(-1),1e-6)
    cx=(weight*AX).sum(-1)/mass; cy=(weight*AY).sum(-1)/mass
    sx=np.sqrt(np.maximum((weight*(AX-cx[...,None])**2).sum(-1)/mass,0))
    sy=np.sqrt(np.maximum((weight*(AY-cy[...,None])**2).sum(-1)/mass,0))
    mean=d.sum(-1)/np.maximum(valid.sum(-1),1)
    previous=pg.reshape(d.shape); common=valid&pm.reshape(valid.shape)
    common_count=common.sum(-1)
    # Compare the SAME spatial support in both frames: merely losing a top row
    # cannot masquerade as downward motion of the weighted center.
    def pair_stats(distances):
        w=np.where(common,(1-distances)**2,0); m=np.maximum(w.sum(-1),1e-6)
        return (w*AX).sum(-1)/m,(w*AY).sum(-1)/m,(distances*common).sum(-1)/np.maximum(common_count,1)
    pcx,pcy,pmean=pair_stats(previous); ncx,ncy,nmean=pair_stats(d)
    previous_count=pm.reshape(valid.shape).sum(-1)
    reliable=(common_count>=MIN_VALID)&(2*common_count>=np.maximum(valid.sum(-1),previous_count))
    dx=np.where(reliable,ncx-pcx,0); dy=np.where(reliable,ncy-pcy,0); dm=np.where(reliable,nmean-pmean,0)
    energy=np.where(reliable,(np.abs(d-previous)*common).sum(-1)/np.maximum(common_count,1),0)
    phys=np.stack((cx,cy,sx,sy,mean,valid.mean(-1),dx,dy,dm,energy,np.cumsum(dx,axis=-1),np.cumsum(dy,axis=-1)),axis=-1)
    local=np.concatenate((gx.reshape(*d.shape[:-1],64),gy.reshape(*d.shape[:-1],64),dt.reshape(*d.shape[:-1],64)),axis=-1)
    return np.concatenate((d,valid.astype(np.float32),local.astype(np.float32),phys.astype(np.float32)),axis=-1)

def load(review_path=REVIEW_PATH):
    candidates=[]; labels_by_hash={}; annotations=load_review_annotations(review_path) if review_path else {}
    for archive in sorted((ROOT/'NN/Data').rglob('*.zip')):
        folder_label=LABEL.get(archive.parent.name)
        if folder_label is None: continue
        split='validation' if archive.stem.endswith('2') else 'train'
        with zipfile.ZipFile(archive) as z:
            for name in sorted(n for n in z.namelist() if n.endswith('frames.csv')):
                label=folder_label
                body=z.read(name); digest=hashlib.sha256(body).hexdigest()
                rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
                # Convert the wire sentinel before narrowing, avoiding int16 overflow.
                raw=np.asarray([[int(r[f'zone_{i:02d}_mm']) for i in range(64)] for r in rows],np.int64)
                if np.any((raw < -1) | ((raw > 32767)&(raw != 65535))):
                    raise ValueError(f'Out-of-range distance: {archive}::{name}')
                raw=np.where(raw==65535,-1,raw).astype(np.int16)
                times=np.asarray([int(r['unix_ms']) for r in rows],np.int64)
                seg=[]; a=0
                for i in range(len(raw)):
                    ok=int((raw[i]>=0).sum())>=MIN_VALID
                    if not ok or (i>a and not GAP_MIN<=times[i]-times[i-1]<=GAP_MAX):
                        if i>a: seg.append((a,i))
                        a=i if ok else i+1
                if a<len(raw): seg.append((a,len(raw)))
                source=str(archive.resolve())+'::'+name
                review=reviewed_for(source,annotations)
                label_spans=label_spans_for(len(raw),label,review,digest)
                if not label_spans: continue
                labels_by_hash.setdefault(digest,set()).add(tuple(label_spans))
                unique_labels={v for _,_,v in label_spans}
                label=next(iter(unique_labels)) if len(unique_labels)==1 else None
                candidates.append(dict(raw=raw,times=times,segments=seg,label=label, label_spans=label_spans,split=split,
                    archive=str(archive.relative_to(ROOT)),member=name,sha256=digest,frames=len(raw)))
    # Conflicting copies carry contradictory labels; exclude the duplicated sequence
    # from both classes instead of letting ZIP sort order select its target.
    records=[r for r in candidates if len(labels_by_hash[r['sha256']])==1]
    conflicts={h for h,labels in labels_by_hash.items() if len(labels)>1}
    print(f'Excluded {len(conflicts)} recordings duplicated across conflicting class folders',flush=True)
    return records

class PhysicsFallNet(nn.Module):
    def __init__(self):
        super().__init__(); self.spatial=nn.Linear(128,PROJECT); self.local=nn.Linear(192,LOCAL)
        self.lstm=nn.LSTM(PROJECT+LOCAL+12,HIDDEN,batch_first=True)
        self.head=nn.Linear(2*HIDDEN,DENSE); self.out=nn.Linear(DENSE,3)
        self.velocity=nn.Linear(HIDDEN,1); self.dropout=nn.Dropout(.2)
    def forward(self,x,lengths,aux=False):
        a=torch.relu(self.spatial(x[:,:,:128])); b=torch.relu(self.local(x[:,:,128:320]))
        z=torch.cat((a,b,x[:,:,320:]),-1)
        pk=pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        pk,(h,_)=self.lstm(pk); seq,_=pad_packed_sequence(pk,batch_first=True,total_length=x.shape[1])
        mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        mx=seq.masked_fill(~mask[:,:,None],-torch.inf).amax(1); last=h[-1]
        logits=self.out(self.dropout(torch.relu(self.head(torch.cat((last,mx),1)))))
        return (logits,self.velocity(last).squeeze(-1)) if aux else logits

def pack(clips):
    lengths=np.array([len(x) for x in clips],np.int64); raw=np.full((len(clips),WINDOW,64),-1,np.int16)
    for i,x in enumerate(clips): raw[i,:len(x)]=x
    return raw,torch.from_numpy(features(raw)),torch.from_numpy(lengths)

def augment(c,rng):
    x=c.copy().reshape(-1,8,8)
    if rng.random()<.5: x=x[:,:,::-1].copy()
    v=x>=0; x=np.where(v,np.clip(x*rng.uniform(.82,1.18)+rng.uniform(-100,100),0,32767),-1).astype(np.int16)
    before=x.copy()
    x[rng.random(x.shape)<.02]=-1
    # Intermittent strips and blocks simulate broken silhouettes, not new falls.
    # Apply to every class; retain at least MIN_VALID measured zones per frame.
    for _ in range(2):
        if rng.random()>=.65: continue
        start=int(rng.integers(len(x))); end=min(len(x),start+int(rng.integers(1,4)))
        y,z=int(rng.integers(8)),int(rng.integers(8)); mode=int(rng.integers(3))
        ys=slice(y,y+1) if mode==0 else slice(None) if mode==1 else slice(y,min(8,y+2))
        zs=slice(None) if mode==0 else slice(z,z+1) if mode==1 else slice(z,min(8,z+2))
        x[start:end,ys,zs]=-1
    unusable=(x>=0).sum(axis=(1,2))<MIN_VALID
    x[unusable]=before[unusable]
    return x.reshape(-1,64)

def class_windows(records,split):
    data={i:[] for i in range(3)}
    for r in records:
        if r['split']!=split: continue
        for sa,sb,label in r['label_spans']:
            for qa,qb in r['segments']:
                a,b=max(sa,qa),min(sb,qb)
                if b<=a or label is None: continue
                seq=r['raw'][a:b]
                if len(seq)<=WINDOW: data[label].append(seq)
                else:
                    starts=list(range(a,b-WINDOW+1,WINDOW//2))
                    starts.append(b-WINDOW)
                    data[label].extend(r['raw'][s:s+WINDOW] for s in sorted(set(starts)))
    return data

def probability(model,clips):
    _,x,l=pack(clips)
    with torch.no_grad(): return model(x,l).softmax(-1).numpy()


def sample_training(train,rng,sampling='shared'):
    clips=[]; labels=[]
    for _ in range(24):
        ids=[rng.integers(len(train[k])) for k in range(3)]
        shared=min(WINDOW,max(6,min(len(train[k][j]) for k,j in enumerate(ids))))
        for k,j in enumerate(ids):
            seq=train[k][j]
            # Short fall clips must not truncate unrelated 24-frame motion/still
            # examples. Keep the complete available context for each sample.
            length=min(len(seq),WINDOW) if sampling=='independent' else min(len(seq),shared)
            start=rng.integers(max(1,len(seq)-length+1))
            clips.append(augment(seq[start:start+length],rng));labels.append(k)
    return clips,labels

def metrics(y,p):
    pred=p.argmax(-1); cm=[[int(((y==i)&(pred==j)).sum()) for j in range(3)] for i in range(3)]
    rec=[cm[i][i]/sum(cm[i]) if sum(cm[i]) else None for i in range(3)]
    precision=[cm[i][i]/max(1,sum(cm[j][i] for j in range(3))) for i in range(3)]
    f1=[2*p*r/max(p+r,1e-12) if r is not None else None for p,r in zip(precision,rec)]
    return dict(labels=['motion','fall','person_still'],confusion_matrix=cm,accuracy=float((pred==y).mean()) if len(y) else None,recall=rec,precision=precision,f1=f1,macro_f1=float(np.mean([f for f in f1 if f is not None])) if len(y) else None,balanced_accuracy=float(np.mean([r for r in rec if r is not None])) if len(y) else None,sequences=len(y))

def numpy_predict(raw,w):
    x=features(raw); h=np.zeros(HIDDEN,np.float32); c=h.copy(); mx=np.full(HIDDEN,-np.inf,np.float32)
    sig=lambda a:1/(1+np.exp(-np.clip(a,-80,80)))
    for frame in x:
        p=np.maximum(w['spatial.weight']@frame[:128]+w['spatial.bias'],0)
        q=np.maximum(w['local.weight']@frame[128:320]+w['local.bias'],0)
        z=np.r_[p,q,frame[320:]]
        g=w['lstm.weight_ih_l0']@z+w['lstm.weight_hh_l0']@h+w['lstm.bias_ih_l0']+w['lstm.bias_hh_l0']
        i,f,n,o=np.split(g,4); c=sig(f)*c+sig(i)*np.tanh(n); h=sig(o)*np.tanh(c); mx=np.maximum(mx,h)
    v=np.r_[h,mx]; d=np.maximum(w['head.weight']@v+w['head.bias'],0)
    logits=w['out.weight']@d+w['out.bias']; q=np.exp(logits-logits.max()); return q/q.sum()

def export(w,vectors,lengths,probs):
    def arr(n,a,typ='float',storage=False):
        v=np.asarray(a).reshape(-1); q=[f'{float(x):.9e}f' if typ=='float' else str(int(x)) for x in v]
        placement='FALL_WEIGHT_STORAGE ' if storage else ''
        return f'static const {typ} {placement}{n}[{len(v)}] = {{\n    '+',\n    '.join(', '.join(q[i:i+8]) for i in range(0,len(q),8))+'\n};\n'
    s='// Generated by NN/train_physics_fall.py; side view, heatmap +Y points downward.\n#pragma once\n#include <stdint.h>\n'
    s+='#ifdef ESP_PLATFORM\n#include "esp_attr.h"\n#define FALL_WEIGHT_STORAGE DRAM_ATTR\n#else\n#define FALL_WEIGHT_STORAGE\n#endif\n'
    for k,v in dict(WINDOW=WINDOW,PROJECT=PROJECT,LOCAL=LOCAL,HIDDEN=HIDDEN,DENSE=DENSE,MIN_VALID=MIN_VALID,GAP_MIN_MS=GAP_MIN,GAP_MAX_MS=GAP_MAX).items(): s+=f'#define FALL_{k} {v}\n'
    s+=f'#define FALL_DISTANCE_SCALE {SCALE:.1f}f\n#define FALL_FEATURE_VERSION {FEATURE_VERSION}\n'
    for k,v in w.items():
        if not k.startswith('velocity.'): s+=arr('fall_'+k.replace('.','_'),v,storage=True)
    s+='#define FALL_SELF_TEST_COUNT 3\n'+arr('fall_test_raw',vectors[:3],'int16_t')+arr('fall_test_lengths',lengths[:3],'uint8_t')+arr('fall_test_probability',probs[:3])
    (ROOT/'main/fall_model_data.h').write_text(s,encoding='utf-8')

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=160); ap.add_argument('--seed',type=int,default=20261028)
    ap.add_argument('--out',type=Path,default=OUT); ap.add_argument('--no-export',action='store_true')
    ap.add_argument('--sampling',choices=['shared','independent'],default='shared')
    args=ap.parse_args(); out=args.out
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed); random.seed(args.seed)
    recs=assign_splits(load())
    train=class_windows(recs,'train'); val=class_windows(recs,'validation')
    for k in (0,1,2):
        if not train[k]: raise ValueError(f'No training sequences for class {k}')
    vc=[(x,k) for k in range(3) for x in val[k]]
    if not vc: raise ValueError('No archive-level validation data')
    vcx=[x for x,k in vc]; vy=np.array([k for x,k in vc]); _,vx,vl=pack(vcx)
    weights=torch.tensor([1/max(1,len(train[i])) for i in range(3)],dtype=torch.float32);weights/=weights.sum(); weights*=3
    model=PhysicsFallNet(); opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.004)
    best=None; bestloss=float('inf'); best_epoch=0; hist=[]
    lengths_seen=[]
    for epoch in range(1,args.epochs+1):
        clips,labels=sample_training(train,rng,args.sampling)
        if epoch==1: lengths_seen=[len(c) for c in clips]
        _,x,l=pack(clips); y=torch.tensor(labels,dtype=torch.long); last=torch.arange(len(l)); feat=x[last,l-1,320:]
        # Auxiliary displacement uses common valid support, not mask-induced
        # changes of the instantaneous weighted center. No acceleration is fitted.
        model.train(); lossvals=[]
        for ix in torch.randperm(len(x)).split(24):
            logits,vhat=model(x[ix],l[ix],aux=True); target=feat[ix,11]
            fall=y[ix]==1; nonfall=~fall
            cls=nn.functional.cross_entropy(logits,y[ix],weight=weights)
            gravity=(torch.relu(.06-vhat[fall]).mean() if fall.any() else logits.sum()*0)
            velocity=nn.functional.smooth_l1_loss(vhat,target)
            loss=cls+.22*velocity+.3*gravity
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();lossvals.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): vlss=float(nn.functional.cross_entropy(model(vx,vl),torch.tensor(vy),weight=weights))
        hist.append(dict(epoch=epoch,train_loss=float(np.mean(lossvals)),validation_loss=vlss))
        if vlss<bestloss-1e-5: bestloss=vlss;best_epoch=epoch;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch%10==0: print(json.dumps(hist[-1]),flush=True)
        if epoch-best_epoch>=25: break
    model.load_state_dict(best);model.eval(); w={k:v.numpy() for k,v in best.items()}
    p=probability(model,vcx)
    full_indices=[i for i,clip in enumerate(vcx) if len(clip)==WINDOW]
    full_validation=metrics(vy[full_indices],p[full_indices])
    full_validation['class_counts']=[int((vy[full_indices]==k).sum()) for k in range(3)]
    full_validation['limitation']='Missing-class recall is undefined; this is window classification, not event evaluation.'
    # No measurement establishes missing observation, never verified vacancy.
    empty_raw=np.full((4,64),-1,np.int16)
    empty_detect=[int((x>=0).sum())==0 for x in empty_raw]
    refs=[next(x for x in train[k] if len(x)>=WINDOW)[:WINDOW] for k in (0,1,2) if any(len(x)>=WINDOW for x in train[k])]
    refs += [x for x in vcx if len(x)>=WINDOW][:12]
    _,_,lengths=pack(refs); raw=np.full((len(refs),WINDOW,64),-1,np.int16)
    for i,x in enumerate(refs):raw[i,:len(x)]=x
    probs=probability(model,refs); err=max(float(np.max(np.abs(numpy_predict(x[:n],w)-q))) for x,n,q in zip(raw,lengths.numpy(),probs))
    assert err<1e-5
    byclass={str(k):{'train_sequences':len(train[k]),'validation_sequences':len(val[k]),'training_records':sum(any(label==k for _,_,label in r['label_spans']) and r['split']=='train' for r in recs),'validation_records':sum(any(label==k for _,_,label in r['label_spans']) and r['split']=='validation' for r in recs)} for k in range(3)}
    validation_falls=sum(r['split']=='validation' and any(label==1 and min(sb,qb)-max(sa,qa)>=WINDOW for sa,sb,label in r['label_spans'] for qa,qb in r['segments']) for r in recs)
    report=dict(seed=args.seed,architecture='local 8x8 spatial/temporal gradient MLP(192,16) + raw/mask spatial MLP(128,16) + 12 physical features -> LSTM(44,24) -> last/max pooling -> Linear(48,16) -> softmax(3)',
        labels={'motion':0,'fall':1,'person_still':2},runtime_labels={'observation_lost':3,'suspected_fall':4},window_frames=WINDOW,window_seconds=WINDOW/10,parameters=sum(v.size for k,v in w.items() if not k.startswith('velocity.')),
        feature_version=FEATURE_VERSION, sampling=args.sampling,
        sampled_full_windows_first_epoch=sum(n==WINDOW for n in lengths_seen),sampled_sequences_first_epoch=len(lengths_seen),
        review_sha256=hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest() if REVIEW_PATH.exists() else None,
        review_path=str(REVIEW_PATH),
        missing_zone_handling=['distance plus validity mask','masked spatial derivative stencils','common-support temporal center changes with overlap gating','accumulated common-support displacement','random and 1-3-frame stripe/block dropout augmentation'],
        mounting='horizontal side-view; heatmap +Y is down',physics_constraints=['common-support vertical displacement','downward direction prior for fall class','auxiliary accumulated displacement head','no vertical image reflection augmentation'],
        validation=metrics(vy,p),full_window_validation=full_validation,class_coverage=byclass,recordings=[{k:v for k,v in r.items() if k not in ('raw','times','segments')} for r in recs],
        empty_sensor_behavior=dict(valid_zone_count=0,output='observation_lost',retains_previous_alert=True,cases=4),
        full_window_validation_falls=validation_falls,
        limitations=['Eight still-person recordings; the longest recording is held out, with correlated windows.','Two motion recordings and three labelled fall acquisition batches.',
          f'Full-window fall validation contains {validation_falls} recordings/windows; clips remain correlated and do not establish independent event recall.','Recording-level labels do not mark exact fall onset; scores do not represent independent event accuracy or safety certification.'],best_epoch=best_epoch,torch_numpy_max_error=err)
    np.savez(out/'weights.npz',**w);torch.save(best,out/'model.pt');np.savez(out/'verification_vectors.npz',raw=raw,lengths=lengths.numpy(),probabilities=probs)
    if not args.no_export: export(w,raw,lengths.numpy(),probs)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'training_history.json').write_text(json.dumps(hist,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
