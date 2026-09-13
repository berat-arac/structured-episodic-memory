from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoints' / 'sem_40runs.json.gz'


def pearson(xs, ys):
    if not xs or len(xs) != len(ys):
        return 0.0
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=sum((x-mx)**2 for x in xs); dy=sum((y-my)**2 for y in ys)
    if dx<=1e-12 or dy<=1e-12: return 0.0
    return num/((dx*dy)**0.5)


def run_one(seed, mode, outcomes_target=100):
    sem=AdaptiveSEMAgent.load_checkpoint(CHECKPOINT, seed=seed, reset_strategy=True)
    sem.set_eval_mode(True)
    sem.set_intent_mode('random')
    if mode=='no_aim':
        sem.aim_weight=0.0
    elif mode!='full':
        raise ValueError(mode)
    env=PongEnv(PongConfig(seed=seed,win_score=999))
    coach=build_coach(seed ^ 0x7788, 220 + (seed % 97))
    pairs=[]; hits=misses=frames=0
    while hits+misses<outcomes_target and frames<260000:
        obs=env.observe(); v=sem.act(obs)
        obs,events=env.step(coach.act(obs),v,1/env.cfg.fps); sem.on_events(obs,events)
        for ev in events:
            if ev['type']=='hit' and ev['side']=='right':
                ep=sem.motor_episodes[-1]
                pairs.append((float(ep['intent']),float(ep['actual_contact'])))
                hits+=1
            elif ev['type']=='score' and ev['side']=='left':
                misses+=1
        frames+=1
    xs=[a for a,b in pairs]; ys=[b for a,b in pairs]
    return {
        'seed':seed,'mode':mode,'pairs':len(pairs),'hits':hits,'misses':misses,
        'hit_rate':hits/max(1,hits+misses),
        'mae':sum(abs(a-b) for a,b in pairs)/max(1,len(pairs)),
        'correlation':pearson(xs,ys),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=8); p.add_argument('--outcomes',type=int,default=100); p.add_argument('--seed',type=int,default=62000); p.add_argument('--report',default='results/aim_ablation.json'); args=p.parse_args()
    modes={'full':[],'no_aim':[]}
    for i in range(args.seeds):
        seed=args.seed+i*137
        for m in modes: modes[m].append(run_one(seed,m,args.outcomes))
        f=modes['full'][-1]; n=modes['no_aim'][-1]
        print(f"seed {seed}: corr full={f['correlation']:.3f} noaim={n['correlation']:.3f} | MAE {f['mae']:.3f}/{n['mae']:.3f} | hit {f['hit_rate']:.3f}/{n['hit_rate']:.3f}")
    agg={}
    for m,rows in modes.items():
        agg[m]={
            'mean_correlation':mean(r['correlation'] for r in rows),
            'mean_mae':mean(r['mae'] for r in rows),
            'mean_hit_rate':mean(r['hit_rate'] for r in rows),
            'mean_pairs':mean(r['pairs'] for r in rows),
        }
    paired={
        'mean_correlation_advantage_full_minus_noaim':mean(f['correlation']-n['correlation'] for f,n in zip(modes['full'],modes['no_aim'])),
        'mean_mae_improvement_noaim_minus_full':mean(n['mae']-f['mae'] for f,n in zip(modes['full'],modes['no_aim'])),
        'full_higher_correlation_seeds':sum(f['correlation']>n['correlation'] for f,n in zip(modes['full'],modes['no_aim'])),
    }
    report={'status':'SEM_AIM_ABLATION_COMPLETE','protocol':{'paired_seeds':True,'seeds':args.seeds,'outcomes_per_seed':args.outcomes,'intent_mode':'random','eval_mode':True,'full':'pretrained aim residual active','no_aim':'same checkpoint and random intents, but aim_weight=0 so intent cannot affect motor action selection'},'aggregate':agg,'paired':paired,'runs':modes}
    out=Path(args.report); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print('report:',out)
if __name__=='__main__': main()
