from __future__ import annotations
import argparse, json, math, random, types
from pathlib import Path
from statistics import mean
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent


def install_mode(sem,mode):
    sem.aim_weight=0.0; sem.set_intent_mode('neutral')
    if mode=='full_sensor': return
    if mode!='no_vertical_velocity': raise ValueError(mode)
    original=sem._features
    def f(self,obs):
        d=original(obs); d['vf']=2; d['vs']=0; return d
    sem._features=types.MethodType(f,sem)


def setup_trial(env, sign, rng):
    c=env.cfg
    env.rally_hits=0
    env.left_y=c.height/2; env.right_y=c.height/2
    env.ball_x=970.0
    env.ball_y=c.height/2 + rng.uniform(-4.0,4.0)
    env.ball_vx=650.0
    env.ball_vy=float(sign)*700.0


def run_one(seed,mode,episodes=320):
    rng=random.Random(seed^0xD1A)
    sem=AdaptiveSEMAgent(seed=seed+9000); install_mode(sem,mode)
    cfg=PongConfig(seed=seed,win_score=999,hit_accel=1.0,max_ball_speed=1100.0)
    env=PongEnv(cfg)
    outcomes=[]; first_actions=[]; frames=0
    for ep in range(episodes):
        sign=-1 if rng.random()<0.5 else 1
        setup_trial(env,sign,rng)
        # Reset only transient actuator/trace state between independent trials;
        # learned tables are retained.
        sem.incoming_active=False; sem.motor_trace.clear(); sem.aim_trace.clear(); sem.decision_frames_left=0; sem.command_action=0.0; sem.applied_velocity=0.0
        got=False; first_action=None
        for _ in range(80):
            obs=env.observe(); sv=sem.act(obs)
            if first_action is None and sem.incoming_active:
                first_action=sem.command_action
            obs,evs=env.step(0.0,sv,1/env.cfg.fps); sem.on_events(obs,evs); frames+=1
            for ev in evs:
                if ev['type']=='hit' and ev['side']=='right': outcomes.append(1); got=True
                elif ev['type']=='score' and ev['side']=='left': outcomes.append(0); got=True
            if got: break
        first_actions.append({'sign':sign,'action':float(first_action or 0.0),'hit':outcomes[-1]})
    first=outcomes[:80]; last=outcomes[-80:]
    tail=first_actions[-80:]
    # Correct first action is same sign as ball vy because + action moves paddle down.
    correct=sum((r['action']>0 and r['sign']>0) or (r['action']<0 and r['sign']<0) for r in tail)/len(tail)
    return {'seed':seed,'mode':mode,'episodes':episodes,'early_hit_rate':sum(first)/len(first),'late_hit_rate':sum(last)/len(last),'early_misses':len(first)-sum(first),'late_misses':len(last)-sum(last),'late_first_action_direction_accuracy':correct,'frames':frames}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=8); p.add_argument('--episodes',type=int,default=320); p.add_argument('--seed',type=int,default=51000); p.add_argument('--report',default='results/direction_ambiguity_benchmark.json'); args=p.parse_args()
    modes={'full_sensor':[],'no_vertical_velocity':[]}
    for i in range(args.seeds):
        seed=args.seed+i*97
        for m in modes:modes[m].append(run_one(seed,m,args.episodes))
        f=modes['full_sensor'][-1]; n=modes['no_vertical_velocity'][-1]
        print(seed,'late hit',round(f['late_hit_rate'],3),round(n['late_hit_rate'],3),'first-dir',round(f['late_first_action_direction_accuracy'],3),round(n['late_first_action_direction_accuracy'],3))
    agg={m:{'mean_early_hit_rate':mean(r['early_hit_rate'] for r in rs),'mean_late_hit_rate':mean(r['late_hit_rate'] for r in rs),'mean_late_misses':mean(r['late_misses'] for r in rs),'mean_late_first_action_direction_accuracy':mean(r['late_first_action_direction_accuracy'] for r in rs)} for m,rs in modes.items()}
    paired={'late_hit_rate_advantage_full_minus_no_vy':mean(f['late_hit_rate']-n['late_hit_rate'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity'])),'first_action_direction_advantage_full_minus_no_vy':mean(f['late_first_action_direction_accuracy']-n['late_first_action_direction_accuracy'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity'])),'full_better_hit_seeds':sum(f['late_hit_rate']>n['late_hit_rate'] for f,n in zip(modes['full_sensor'],modes['no_vertical_velocity']))}
    report={'status':'SEM_DIRECTION_AMBIGUITY_BENCHMARK_COMPLETE','protocol':{'paired_seeds':True,'seeds':args.seeds,'episodes':args.episodes,'initial_geometry':'same x/y/paddle geometry, ball_vy sign randomized ±700, vx=650, close to SEM paddle','aim_weight':0.0,'reward':'sparse hit/miss only','full_sensor':'original state','no_vertical_velocity':'vertical velocity bins neutralized','note':'Designed so the first motor decision occurs before position alone reveals the trajectory direction.'},'aggregate':agg,'paired':paired,'runs':modes}
    out=ROOT/args.report; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print(out)
if __name__=='__main__': main()
