from __future__ import annotations
import argparse, json
from pathlib import Path
from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent

class SpecificOnlySEM(AdaptiveSEMAgent):
    def _base_motor_keys(self, obs):
        keys=super()._base_motor_keys(obs)
        return (keys[0],)

class CoarseOnlySEM(AdaptiveSEMAgent):
    def _base_motor_keys(self, obs):
        keys=super()._base_motor_keys(obs)
        return tuple(keys[1:])


def used_key(table,key):
    return key in table.table and any(s.n>0 for s in table.table[key].values())

def reset_no_learning(sem,events):
    if any((e['type']=='hit' and e['side']=='right') or (e['type']=='score' and e['side']=='left') for e in events):
        sem.motor_trace.clear(); sem.aim_trace.clear(); sem.incoming_decision_audit.clear(); sem.incoming_active=False; sem.decision_frames_left=0; sem.pending_shot=None

def cfg(seed,shift):
    if shift=='id': return PongConfig(seed=seed,win_score=999)
    return PongConfig(seed=seed,win_score=999,height=900,paddle_h=65,sem_max_speed=850.0,
                      ball_base_speed=455.0,hit_accel=1.07,max_ball_speed=1380.0,bounce_angle_scale=1.52)

def play(cls,ckpt,seed,idx,shift,n):
    sem=cls.load_checkpoint(ckpt,seed=seed^0x3551,reset_strategy=True)
    sem.set_eval_mode(True); sem.set_intent_mode('neutral'); sem.aim_weight=0.0
    env=PongEnv(cfg(seed,shift)); coach=build_coach(seed^0x7189,idx)
    out=[]; frames=0; decisions=0; specific_seen=0; unseen_outcomes=[]; current_had_unseen=False
    while len(out)<n and frames<250000:
        obs=env.observe()
        if obs['ball_vx']>0 and (not sem.incoming_active or sem.decision_frames_left<=0):
            # Always inspect the original most-specific key independent of ablation class.
            orig=AdaptiveSEMAgent._base_motor_keys(sem,obs)[0]
            seen=used_key(sem.motor,orig)
            decisions+=1; specific_seen+=int(seen)
            if not seen: current_had_unseen=True
        v=sem.act(obs); obs,events=env.step(coach.act(obs),v,1/env.cfg.fps)
        terminal=None
        for e in events:
            if e['type']=='hit' and e['side']=='right': terminal=1
            elif e['type']=='score' and e['side']=='left': terminal=0
        if terminal is not None:
            out.append(terminal)
            if current_had_unseen: unseen_outcomes.append(terminal)
            current_had_unseen=False
        reset_no_learning(sem,events); frames+=1
    return {'hit_rate':sum(out)/max(1,len(out)),'specific_seen_rate':specific_seen/max(1,decisions),
            'unseen_episode_count':len(unseen_outcomes),'unseen_episode_hit_rate':sum(unseen_outcomes)/max(1,len(unseen_outcomes)),
            'outcomes':len(out)}

def mean(rows,key): return sum(x[key] for x in rows)/max(1,len(rows))

def run(args):
    ckpt=Path(args.checkpoint); ckpt=ckpt if ckpt.is_absolute() else ROOT/ckpt
    variants={'full_overlap':AdaptiveSEMAgent,'specific_only':SpecificOnlySEM,'coarse_only':CoarseOnlySEM}
    report={}
    shifts=('id','hard_compound') if not args.shift else (args.shift,)
    chosen=variants if not args.variant else {args.variant: variants[args.variant]}
    for shift in shifts:
        report[shift]={}
        for name,cls in chosen.items():
            rows=[]
            for i in range(args.seeds):
                seed=args.base_seed+i*211
                rows.append(play(cls,ckpt,seed,i,shift,args.outcomes))
            report[shift][name]={'rows':rows,'mean_hit_rate':mean(rows,'hit_rate'),
                                 'mean_specific_seen_rate':mean(rows,'specific_seen_rate'),
                                 'mean_unseen_episode_hit_rate':mean(rows,'unseen_episode_hit_rate'),
                                 'mean_unseen_episode_count':mean(rows,'unseen_episode_count')}
            x=report[shift][name]
            print(f"{shift:13s} {name:14s} hit={x['mean_hit_rate']:.3f} unseenHit={x['mean_unseen_episode_hit_rate']:.3f} specificSeen={x['mean_specific_seen_rate']:.2f}")
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True)
    payload={'status':'ABSTRACTION_TRANSFER_ABLATION_COMPLETE','seeds':args.seeds,'outcomes':args.outcomes,
             'aim_weight':0.0,'learning_frozen':True,'report':report}
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('report:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz'); p.add_argument('--base-seed',type=int,default=20270301)
    p.add_argument('--seeds',type=int,default=6); p.add_argument('--outcomes',type=int,default=70); p.add_argument('--shift',choices=['','id','hard_compound'],default=''); p.add_argument('--variant',choices=['','full_overlap','specific_only','coarse_only'],default=''); p.add_argument('--output',default='results/abstraction_transfer_ablation.json')
    run(p.parse_args())
