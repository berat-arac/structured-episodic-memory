from __future__ import annotations
import argparse, json, math, types
from pathlib import Path
from statistics import mean
from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent

def install_motor_mode(sem, mode):
    sem.aim_weight = 0.0
    if mode == 'full':
        return
    if mode != 'fast_only':
        raise ValueError(mode)
    def fast_estimate(self, keys, action):
        weighted=0.0; weight_sum=0.0; evidence=0; disagreement=0.0
        for level,key in enumerate(keys):
            stat=self.table[key][action]
            specificity=1.0/(1.0+0.20*level)
            confidence=min(1.0,stat.n/7.0)
            w=specificity*(0.25+0.75*confidence)
            value=stat.fast
            weighted += w*value; weight_sum += w; evidence += stat.n
            disagreement=max(disagreement,abs(stat.fast-stat.slow))
        return weighted/max(weight_sum,1e-9), evidence, disagreement
    sem.motor._estimate = types.MethodType(fast_estimate, sem.motor)

def latency(seq, window=20, threshold=.75):
    for end in range(window,len(seq)+1):
        if sum(seq[end-window:end])/window >= threshold:
            return end
    return None

def phase(sem,seed,idx,n,invert):
    env=PongEnv(PongConfig(seed=seed,win_score=999)); coach=build_coach(seed^0x2A77,900+idx)
    seq=[]; frames=0
    while len(seq)<n and frames<400000:
        obs=env.observe(); cmd=sem.act(obs); applied=-cmd if invert else cmd
        obs,evs=env.step(coach.act(obs),applied,1/env.cfg.fps); sem.on_events(obs,evs)
        for e in evs:
            if e['type']=='hit' and e['side']=='right': seq.append(1)
            elif e['type']=='score' and e['side']=='left': seq.append(0)
        frames+=1
    return {'early20':sum(seq[:20])/20,'first50':sum(seq[:50])/50,'first100':sum(seq[:100])/100,
            'late20':sum(seq[-20:])/20,'rate':sum(seq)/len(seq),'lat75':latency(seq),'sequence':seq}

def agent(cp,seed,mode):
    s=AdaptiveSEMAgent.load_checkpoint(cp,seed=seed^0x7731,reset_strategy=True)
    s.set_intent_mode('neutral'); s.set_eval_mode(False); install_motor_mode(s,mode); return s

def run(args):
    cp=Path(args.checkpoint); cp=cp if cp.is_absolute() else ROOT/cp
    modes={'full':[],'fast_only':[]}
    for i in range(args.seeds):
        seed=args.base_seed+i*317
        for mode in modes:
            s=agent(cp,seed,mode)
            a1=phase(s,seed,i,args.a1,False); b1=phase(s,seed,i,args.b,True); a2=phase(s,seed,i,args.a2,False); b2=phase(s,seed,i,args.b,True)
            modes[mode].append({'seed':seed,'A1':a1,'B1':b1,'A2':a2,'B2':b2})
        f=modes['full'][-1]; g=modes['fast_only'][-1]
        print(seed,'full B1/B2',f['B1']['lat75'],f['B2']['lat75'],'fast',g['B1']['lat75'],g['B2']['lat75'],
              'first100 B2',f['B2']['first100'],g['B2']['first100'])
    def ag(rows):
        out={}
        for ph in ['A1','B1','A2','B2']:
            out[ph]={k:mean(r[ph][k] for r in rows if r[ph][k] is not None) for k in ['early20','first50','first100','late20','rate']}
            ls=[r[ph]['lat75'] for r in rows if r[ph]['lat75'] is not None]
            out[ph]['lat75_mean']=mean(ls) if ls else None; out[ph]['lat75_adapted']=len(ls); out[ph]['lat75_values']=[r[ph]['lat75'] for r in rows]
        out['savings']={
          'mean_B1_minus_B2_latency_among_paired': mean(r['B1']['lat75']-r['B2']['lat75'] for r in rows if r['B1']['lat75'] is not None and r['B2']['lat75'] is not None),
          'mean_B2_minus_B1_first100': mean(r['B2']['first100']-r['B1']['first100'] for r in rows),
        }
        return out
    aggregate={m:ag(r) for m,r in modes.items()}
    paired=[]
    for f,g in zip(modes['full'],modes['fast_only']):
        paired.append({'seed':f['seed'],
          'full_B2_first100':f['B2']['first100'],'fast_B2_first100':g['B2']['first100'],
          'full_B2_advantage_first100':f['B2']['first100']-g['B2']['first100'],
          'full_B2_lat75':f['B2']['lat75'],'fast_B2_lat75':g['B2']['lat75']})
    report={'status':'MOTOR_RETENTION_MEMORY_ABLATION_COMPLETE','protocol':{
      'full':'original survival motor 0.70 fast + 0.30 slow','fast_only':'slow estimate ignored in survival action selection; updates still occur',
      'aim_weight':0.0,'phases':'A1 normal -> B1 inverted -> A2 normal -> B2 inverted','switch_signaled':False,'memory_reset':False},
      'aggregate':aggregate,'paired':paired,'runs':modes}
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'paired':paired},indent=2)); print('report',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz'); p.add_argument('--base-seed',type=int,default=20270601)
    p.add_argument('--seeds',type=int,default=2); p.add_argument('--a1',type=int,default=100); p.add_argument('--b',type=int,default=220); p.add_argument('--a2',type=int,default=220)
    p.add_argument('--output',default='results/motor_retention_memory_ablation.json'); run(p.parse_args())
