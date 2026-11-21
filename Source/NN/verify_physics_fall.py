"""Verify the ESP32 C implementation against the three-state PyTorch model."""
import ctypes as ct,json,subprocess
from pathlib import Path
import numpy as np,torch
from train_physics_fall import ROOT,PhysicsFallNet,WINDOW,load,pack,probability,assign_splits

OUT=ROOT/'NN/artifacts/physics_fall'
class State(ct.Structure):
    _fields_=[('frames',(ct.c_int16*64)*WINDOW),('last_ms',ct.c_uint32),('count',ct.c_uint8),('next',ct.c_uint8),('high',ct.c_uint8),('low',ct.c_uint8),('predict_phase',ct.c_uint8),('stable',ct.c_int8),('raw',ct.c_int8),('has_time',ct.c_bool),('probabilities',ct.c_float*3),
        ('motion_history',ct.c_uint32),('downward_history',ct.c_uint32),('last_motion_ms',ct.c_uint32),('last_downward_ms',ct.c_uint32),('last_candidate_ms',ct.c_uint32),
        ('has_motion',ct.c_bool),('has_downward',ct.c_bool),('has_candidate',ct.c_bool),('observation_lost',ct.c_bool),('held_alert',ct.c_int8)]
class Result(ct.Structure):
    _fields_=[('raw',ct.c_int8),('stable',ct.c_int8),('valid',ct.c_uint8),('buffered',ct.c_uint8),('p',ct.c_float*3),('us',ct.c_uint32),('updated',ct.c_bool)]

def main():
    subprocess.run(['gcc','-O3','-Wall','-Wextra','-Werror','-std=c11','-shared','-I',str(ROOT/'main'),str(ROOT/'main/fall_classifier.c'),str(ROOT/'tests/fall_test_bridge.c'),'-o',str(OUT/'physics_classifier.dll'),'-lm'],check=True)
    lib=ct.CDLL(str(OUT/'physics_classifier.dll')); lib.fall_predict_sequence.argtypes=[ct.c_void_p,ct.c_uint,ct.c_void_p];lib.fall_predict_sequence.restype=ct.c_float
    assert ct.sizeof(State)==lib.fall_state_size() and ct.sizeof(Result)==lib.fall_result_size()
    lib.fall_features_for_test.argtypes=[ct.c_void_p,ct.c_uint,ct.c_void_p]
    lib.fall_push.argtypes=[ct.POINTER(State),ct.c_void_p,ct.c_uint32,ct.POINTER(Result)];lib.fall_reset.argtypes=[ct.POINTER(State)]
    lib.fall_self_test.argtypes=[ct.POINTER(ct.c_float)];lib.fall_self_test.restype=ct.c_bool
    lib.fall_encode_for_test.argtypes=[ct.c_void_p,ct.c_uint16,ct.POINTER(Result)]
    torch.set_num_threads(2); model=PhysicsFallNet();w=dict(np.load(OUT/'weights.npz'));model.load_state_dict({k:torch.from_numpy(v) for k,v in w.items()});model.eval()
    vec=np.load(OUT/'verification_vectors.npz');err=ct.c_float();selftest=lib.fall_self_test(ct.byref(err))
    referr=0
    for raw,n,p in zip(vec['raw'],vec['lengths'],vec['probabilities']):
        q=(ct.c_float*3)();lib.fall_predict_sequence(raw.ctypes.data,int(n),q);referr=max(referr,float(np.max(np.abs(np.asarray(q)-p))))
    feature_error=0
    from train_physics_fall import features
    patch=np.full((WINDOW,8,8),-1,np.int16);patch[:,1:7,1:7]=1200
    patch[5:8,3,1:7]=-1;patch[8:11,1:3,1:7]=-1
    rng=np.random.default_rng(105)
    random_frames=rng.integers(0,4500,(WINDOW,64),dtype=np.int16)
    random_frames[rng.random(random_frames.shape)<.4]=-1
    feature_cases=[(raw,int(n)) for raw,n in zip(vec['raw'][:3],vec['lengths'][:3])]
    feature_cases += [(patch.reshape(WINDOW,64),WINDOW),(random_frames,WINDOW),(np.full((WINDOW,64),-1,np.int16),WINDOW)]
    for raw,n in feature_cases:
        cfeat=np.zeros(int(n)*332,np.float32);lib.fall_features_for_test(raw.ctypes.data,int(n),cfeat.ctypes.data)
        pyfeat=features(raw[None,:int(n),:])[0]
        feature_error=max(feature_error,float(np.max(np.abs(cfeat.reshape(int(n),332)-pyfeat))))
    print('selftest',selftest,'selftest_error',err.value,'probability_error',referr,'feature_error',feature_error,flush=True)
    assert referr<1e-5 and feature_error<1e-5 and selftest
    records=assign_splits(load()); streamerr=0; windows=0; occupancy=0; reused=0
    for r in records:
        state=State();lib.fall_reset(ct.byref(state)); result=Result()
        raw=r['raw'];times=r['times']
        expected={}
        if r['label'] is not None:
            clips=[]
            for a,b in r['segments']:
                if b-a>=WINDOW:
                    clips.extend(raw[e-WINDOW:e] for e in range(a+WINDOW,b+1))
            if clips:
                p=probability(model,clips); expected={i:p[j] for j,i in enumerate([e-1 for a,b in r['segments'] if b-a>=WINDOW for e in range(a+WINDOW,b+1)])}
        for i,frame in enumerate(raw):
            lib.fall_push(ct.byref(state),frame.ctypes.data,int(times[i])&0xffffffff,ct.byref(result))
            valid=int((frame>=0).sum())
            if valid==0:
                assert result.raw==3 and result.stable in (1,3,4) and result.buffered==0;occupancy+=1
            elif i in expected:
                assert result.raw in (0,1,2) and result.buffered==WINDOW
                if result.updated:
                    streamerr=max(streamerr,float(np.max(np.abs(np.asarray(result.p)-expected[i]))));windows+=1
                else:
                    assert i-1 in expected
                    streamerr=max(streamerr,float(np.max(np.abs(np.asarray(result.p)-expected[i-1]))));reused+=1
    frame=np.full(64,-1,np.int16);state=State();result=Result();lib.fall_reset(ct.byref(state));wire=[]
    for t in range(4):lib.fall_push(ct.byref(state),frame.ctypes.data,t*100,ct.byref(result));assert result.raw==3 and result.stable==3
    packet=(ct.c_uint8*20)();lib.fall_encode_for_test(packet,0,ct.byref(result));wire.append(bytes(packet).hex())
    frame=np.full(64,1200,np.int16);lib.fall_reset(ct.byref(state))
    for t in range(WINDOW):
        lib.fall_push(ct.byref(state),frame.ctypes.data,1000+t*100,ct.byref(result))
        if t in (0,WINDOW-1):
            packet=(ct.c_uint8*20)();lib.fall_encode_for_test(packet,65535,ct.byref(result));wire.append(bytes(packet).hex())
    (OUT/'packet_vectors.txt').write_text('\n'.join(wire)+'\n',encoding='ascii')
    empty=np.full(64,1000,np.int16)
    for t in range(WINDOW+2):lib.fall_push(ct.byref(state),empty.ctypes.data,1000+t*100,ct.byref(result))
    assert result.buffered==WINDOW and result.raw in range(3) and result.updated
    assert streamerr<1e-5
    summary=dict(passed=True,self_test_error=err.value,feature_max_error=feature_error,feature_cases=len(feature_cases),vector_count=len(vec['raw']),vector_max_error=referr,
        real_stream_windows=windows,real_stream_reused_frames=reused,real_stream_max_error=streamerr,empty_frames=occupancy,missing_behavior='valid_zone_count < 4 => observation lost; existing alerts retained; never proof of vacancy',
        edge_tests=[f'{WINDOW}-frame warmup','ring wrap','counter rollover via uint32 timer','missing observation without vacancy claim','nonempty reacquisition'],
        note='C parity does not establish field fall recall; see full_window_validation_falls in metrics.json.')
    (OUT/'c_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
