from __future__ import annotations

import argparse, json
from pathlib import Path
from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT/'checkpoints'/'sem_40runs.json.gz'


def used_key(table, key):
    if key not in table.table:
        return False
    return any(s.n > 0 for s in table.table[key].values())


def reset_no_learning(sem, events):
    if any((e['type']=='hit' and e['side']=='right') or (e['type']=='score' and e['side']=='left') for e in events):
        sem.motor_trace.clear(); sem.aim_trace.clear(); sem.incoming_decision_audit.clear()
        sem.incoming_active=False; sem.decision_frames_left=0; sem.pending_shot=None


def configs(seed):
    common=dict(seed=seed,win_score=999)
    return {
      'in_distribution': PongConfig(**common),
      'tiny_paddle': PongConfig(**common,paddle_h=55),
      'very_fast_ball': PongConfig(**common,ball_base_speed=500.0,hit_accel=1.075,max_ball_speed=1450.0),
      'very_steep_bounce': PongConfig(**common,bounce_angle_scale=1.68),
      'very_tall_field': PongConfig(**common,height=980),
      'slow_actuator': PongConfig(**common,sem_max_speed=720.0),
      'hard_compound': PongConfig(**common,height=900,paddle_h=65,sem_max_speed=850.0,
                                  ball_base_speed=455.0,hit_accel=1.07,max_ball_speed=1380.0,
                                  bounce_angle_scale=1.52),
    }


def play(ckpt,cfg,seed,idx,n,adaptive):
    sem=AdaptiveSEMAgent.load_checkpoint(ckpt,seed=seed^0x53A1,reset_strategy=True)
    sem.set_intent_mode('neutral'); sem.set_eval_mode(True)
    env=PongEnv(cfg); coach=build_coach(seed^0x771B,700+idx)
    outcomes=[]; frames=0
    decisions=0; exact_seen=0; any_seen=0; exact_unseen_but_coarse_seen=0
    while len(outcomes)<n and frames<260000:
        obs=env.observe()
        if obs['ball_vx']>0 and (not sem.incoming_active or sem.decision_frames_left<=0):
            keys=sem._base_motor_keys(obs)
            flags=[used_key(sem.motor,k) for k in keys]
            decisions+=1
            exact_seen += int(flags[0])
            any_seen += int(any(flags))
            exact_unseen_but_coarse_seen += int((not flags[0]) and any(flags[1:]))
        v=sem.act(obs)
        obs,events=env.step(coach.act(obs),v,1/cfg.fps)
        for e in events:
            if e['type']=='hit' and e['side']=='right': outcomes.append(1)
            elif e['type']=='score' and e['side']=='left': outcomes.append(0)
        if adaptive: sem.on_events(obs,events)
        else: reset_no_learning(sem,events)
        frames+=1
    k=min(20,max(1,len(outcomes)//3)); first=outcomes[:k]; last=outcomes[-k:]
    return {
      'seed':seed,'coach_style':coach.style,'adaptive':adaptive,'outcomes':len(outcomes),
      'hit_rate':sum(outcomes)/max(1,len(outcomes)),
      'early_hit_rate':sum(first)/max(1,len(first)),'late_hit_rate':sum(last)/max(1,len(last)),
      'early_misses':len(first)-sum(first),'late_misses':len(last)-sum(last),
      'decision_count':decisions,
      'exact_specific_key_seen_rate':exact_seen/max(1,decisions),
      'any_abstraction_seen_rate':any_seen/max(1,decisions),
      'exact_unseen_but_coarse_seen_rate':exact_unseen_but_coarse_seen/max(1,decisions),
    }


def mean(rows,key): return sum(r[key] for r in rows)/max(1,len(rows))

def run(args):
    ckpt=Path(args.checkpoint); ckpt=ckpt if ckpt.is_absolute() else ROOT/ckpt
    result={}
    names=list(configs(args.base_seed))
    if getattr(args, 'only', ''):
        requested=[x.strip() for x in args.only.split(',') if x.strip()]
        unknown=[x for x in requested if x not in names]
        if unknown: raise ValueError(f'unknown conditions: {unknown}')
        names=requested
    for ci,name in enumerate(names):
        fr=[]; ad=[]
        for i in range(args.seeds):
            seed=args.base_seed+i*211
            cfg=configs(seed)[name]
            # Paired protocol: identical RNG seed and coach index across all physics conditions.
            fr.append(play(ckpt,cfg,seed,i,args.outcomes,False))
            ad.append(play(ckpt,cfg,seed,i,args.outcomes,True))
        result[name]={
          'frozen':fr,'adaptive':ad,
          'zero_shot_hit_rate':mean(fr,'hit_rate'),
          'adaptive_hit_rate':mean(ad,'hit_rate'),
          'adaptive_early_hit_rate':mean(ad,'early_hit_rate'),
          'adaptive_late_hit_rate':mean(ad,'late_hit_rate'),
          'recovery_delta':mean(ad,'late_hit_rate')-mean(ad,'early_hit_rate'),
          'zero_shot_exact_specific_key_seen_rate':mean(fr,'exact_specific_key_seen_rate'),
          'zero_shot_any_abstraction_seen_rate':mean(fr,'any_abstraction_seen_rate'),
          'zero_shot_exact_unseen_but_coarse_seen_rate':mean(fr,'exact_unseen_but_coarse_seen_rate'),
        }
        x=result[name]
        print(f"{name:18s} zshot={x['zero_shot_hit_rate']:.3f} adapt={x['adaptive_hit_rate']:.3f} "
              f"early={x['adaptive_early_hit_rate']:.3f} late={x['adaptive_late_hit_rate']:.3f} "
              f"exactSeen={x['zero_shot_exact_specific_key_seen_rate']:.2f} anySeen={x['zero_shot_any_abstraction_seen_rate']:.2f} "
              f"coarseTransfer={x['zero_shot_exact_unseen_but_coarse_seen_rate']:.2f}")
    report={'status':'HARD_PHYSICS_GENERALIZATION_COMPLETE','checkpoint':str(ckpt),'seeds':args.seeds,
            'outcomes':args.outcomes,'protocol':{'frozen_learning':True,'exploration_off':True,
            'adaptive_updates':True,'shift_signaled_to_sem':False,'neutral_intent':True},'conditions':result}
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print('report:',out)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz');
    p.add_argument('--base-seed',type=int,default=20270201); p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--outcomes',type=int,default=60); p.add_argument('--only',default='')
    p.add_argument('--output',default='results/hard_physics_smoke.json')
    run(p.parse_args())
