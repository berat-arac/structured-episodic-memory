from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def rolling_latency(seq, window=20, threshold=0.75):
    if len(seq) < window:
        return None
    for end in range(window, len(seq)+1):
        if sum(seq[end-window:end]) / window >= threshold:
            return end
    return None


def phase(sem, seed, run_index, n, invert):
    env = PongEnv(PongConfig(seed=seed, win_score=999))
    coach = build_coach(seed ^ 0x2A77, 900 + run_index)
    seq=[]; frames=0
    while len(seq)<n and frames<400000:
        obs=env.observe(); cmd=sem.act(obs); applied=-cmd if invert else cmd
        obs,events=env.step(coach.act(obs),applied,1/env.cfg.fps); sem.on_events(obs,events)
        for e in events:
            if e['type']=='hit' and e['side']=='right': seq.append(1)
            elif e['type']=='score' and e['side']=='left': seq.append(0)
        frames+=1
    k=min(20,max(1,len(seq)//4))
    return {
        'early':sum(seq[:k])/k,'late':sum(seq[-k:])/k,'rate':sum(seq)/len(seq),
        'lat75':rolling_latency(seq,20,0.75),'lat85':rolling_latency(seq,20,0.85),'sequence':seq
    }


def make_agent(ckpt, seed):
    s=AdaptiveSEMAgent.load_checkpoint(ckpt,seed=seed^0x7731,reset_strategy=True)
    s.set_intent_mode('neutral'); s.set_eval_mode(False); return s


def run(args):
    ckpt=Path(args.checkpoint); ckpt=ckpt if ckpt.is_absolute() else ROOT/ckpt
    rows=[]
    for i in range(args.seeds):
        seed=args.base_seed+i*317
        # Retention path: prior exposure to inverted mapping.
        r=make_agent(ckpt,seed)
        r_a1=phase(r,seed,i,args.a1,False)
        r_b1=phase(r,seed,i,args.b,True)
        r_a2=phase(r,seed,i,args.a2,False)
        r_b2=phase(r,seed,i,args.b,True)

        # Matched-experience control: same number of pre-target outcomes, but all
        # normal mapping; it has never experienced inversion before target B.
        c=make_agent(ckpt,seed)
        c_a1=phase(c,seed,i,args.a1,False)
        c_n1=phase(c,seed,i,args.b,False)
        c_n2=phase(c,seed,i,args.a2,False)
        c_blate=phase(c,seed,i,args.b,True)

        row={'seed':seed,'retention':{'A1':r_a1,'B1':r_b1,'A2':r_a2,'B2':r_b2},
             'matched_control':{'A1':c_a1,'N1':c_n1,'N2':c_n2,'B_late_first_exposure':c_blate}}
        rows.append(row)
        print(f"seed {seed}: B1={r_b1['lat75']} B2={r_b2['lat75']} control-late-B={c_blate['lat75']} | "
              f"early B2={r_b2['early']:.2f} ctl={c_blate['early']:.2f}")

    comp=[]
    for x in rows:
        b2=x['retention']['B2']; ctl=x['matched_control']['B_late_first_exposure']; b1=x['retention']['B1']
        comp.append({'seed':x['seed'],'B1_lat75':b1['lat75'],'B2_lat75':b2['lat75'],'control_lat75':ctl['lat75'],
                     'B2_vs_control_faster': b2['lat75'] is not None and ctl['lat75'] is not None and b2['lat75']<ctl['lat75'],
                     'B2_early':b2['early'],'control_early':ctl['early'],'early_advantage':b2['early']-ctl['early'],
                     'B2_late':b2['late'],'control_late':ctl['late']})
    comparable=[x for x in comp if x['B2_lat75'] is not None and x['control_lat75'] is not None]
    summary={
      'comparable_runs':len(comparable),
      'B2_faster_than_matched_control_runs':sum(x['B2_vs_control_faster'] for x in comparable),
      'mean_B1_lat75':mean(x['B1_lat75'] for x in comp if x['B1_lat75'] is not None),
      'mean_B2_lat75':mean(x['B2_lat75'] for x in comp if x['B2_lat75'] is not None),
      'mean_matched_control_first_B_lat75':mean(x['control_lat75'] for x in comp if x['control_lat75'] is not None),
      'mean_B2_early':mean(x['B2_early'] for x in comp),
      'mean_control_early':mean(x['control_early'] for x in comp),
      'mean_early_advantage':mean(x['early_advantage'] for x in comp),
      'mean_B2_late':mean(x['B2_late'] for x in comp),
      'mean_control_late':mean(x['control_late'] for x in comp),
      'paired':comp,
    }
    payload={'status':'MOTOR_REMAP_RETENTION_MATCHED_CONTROL_COMPLETE','protocol':{
      'retention_path':'A1 normal -> B1 inverted -> A2 normal -> B2 inverted',
      'matched_control':'A1 normal -> N1 normal -> N2 normal -> first inverted B',
      'pre_target_outcomes_matched':True,'same_seed_coach_stream_each_phase':True,
      'switch_signaled_to_sem':False,'memory_reset_between_phases':False,
      'interpretation':'B2 advantage over matched late first-B is evidence of mapping-specific savings beyond generic extra experience.'},
      'summary':summary,'rows':rows}
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('report:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz')
    p.add_argument('--base-seed',type=int,default=20270601); p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--a1',type=int,default=100); p.add_argument('--b',type=int,default=220); p.add_argument('--a2',type=int,default=220)
    p.add_argument('--output',default='results/motor_remap_retention_control.json'); run(p.parse_args())
