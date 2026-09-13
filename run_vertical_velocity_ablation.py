from __future__ import annotations

import argparse, json, types
from pathlib import Path
from statistics import mean

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent


def install_mode(sem,mode):
    sem.aim_weight=0.0
    sem.set_intent_mode('neutral')
    if mode=='full_sensor': return
    if mode!='no_vertical_velocity': raise ValueError(mode)
    original=sem._features
    def no_vy_features(self,obs):
        f=original(obs)
        # Preserve every other sensory/state feature; remove only current
        # vertical ball velocity magnitude/sign from the motor representation.
        f['vf']=2  # bin corresponding to approximately zero normalized vy
        f['vs']=0
        return f
    sem._features=types.MethodType(no_vy_features,sem)


def run_one(seed,mode,outcomes_target=140):
    sem=AdaptiveSEMAgent(seed=seed+10000); install_mode(sem,mode)
    env=PongEnv(PongConfig(seed=seed,win_score=999)); coach=build_coach(seed^0x4455,seed%23)
    outcomes=[]; frames=0
    while len(outcomes)<outcomes_target and frames<260000:
        obs=env.observe(); sv=sem.act(obs); obs,evs=env.step(coach.act(obs),sv,1/env.cfg.fps); sem.on_events(obs,evs)
        for ev in evs:
            if ev['type']=='hit' and ev['side']=='right': outcomes.append(1)
            elif ev['type']=='score' and ev['side']=='left': outcomes.append(0)
        frames+=1
    first=outcomes[:40]; last=outcomes[-40:]
    return {'seed':seed,'mode':mode,'coach_style':coach.style,'outcomes':len(outcomes),'frames':frames,'early_misses':len(first)-sum(first),'late_misses':len(last)-sum(last),'early_hit_rate':sum(first)/max(1,len(first)),'late_hit_rate':sum(last)/max(1,len(last)),'motor_updates':sem.motor.total_updates}


def summarize(rows):
    e=mean(r['early_misses'] for r in rows); l=mean(r['late_misses'] for r in rows)
    return {'mean_early_misses':e,'mean_late_misses':l,'miss_reduction_fraction':(e-l)/max(e,1e-9),'mean_early_hit_rate':mean(r['early_hit_rate'] for r in rows),'mean_late_hit_rate':mean(r['late_hit_rate'] for r in rows),'improved_seeds':sum(r['late_misses']<r['early_misses'] for r in rows),'worse_seeds':sum(r['late_misses']>r['early_misses'] for r in rows)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=8); p.add_argument('--outcomes',type=int,default=140); p.add_argument('--seed',type=int,default=41000); p.add_argument('--report',default='results/vertical_velocity_ablation.json'); args=p.parse_args()
    modes={'full_sensor':[],'no_vertical_velocity':[]}
    for i in range(args.seeds):
        seed=args.seed+i*97
        for m in modes: modes[m].append(run_one(seed,m,args.outcomes))
        f=modes['full_sensor'][-1]; n=modes['no_vertical_velocity'][-1]
        print(seed,'full',f['early_misses'],'->',f['late_misses'],'no-vy',n['early_misses'],'->',n['late_misses'])
    agg={m:summarize(rs) for m,rs in modes.items()}
    paired={'mean_late_miss_penalty_no_vy_minus_full':mean(n['late_misses']-f['late_misses'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity'])),'full_better_late_seeds':sum(f['late_misses']<n['late_misses'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity'])),'ties':sum(f['late_misses']==n['late_misses'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity'])),'no_vy_better_late_seeds':sum(f['late_misses']>n['late_misses'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity']))}
    report={'status':'SEM_VERTICAL_VELOCITY_SENSOR_ABLATION_COMPLETE','protocol':{'paired_seeds':True,'seeds':args.seeds,'outcomes_per_seed':args.outcomes,'aim_weight':0.0,'intent_mode':'neutral','full_sensor':'original motor state','no_vertical_velocity':'vf and vs neutralized; all other state features unchanged','dense_frame_reward':False,'no_action_rule_added':True},'aggregate':agg,'paired':paired,'runs':modes}
    out=ROOT/args.report; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print(out)
if __name__=='__main__': main()
