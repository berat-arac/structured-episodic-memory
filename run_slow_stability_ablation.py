from __future__ import annotations

import argparse, json, math, types
from pathlib import Path
from statistics import mean

import run_opponent_switch_test as sw
from sem_agent import AdaptiveSEMAgent
from pong_core import PongConfig, PongEnv

ROOT=Path(__file__).resolve().parent


def install_mode(sem, mode):
    if mode=='full': return
    if mode!='fast_only': raise ValueError(mode)
    def fast_only_value(self, ctx, action):
        g=self.global_stats[action]; c=self.ctx_stats[ctx][action]
        cw=min(0.72,c.n/7.0)
        base=(1-cw)*g.fast+cw*c.fast
        total_n=sum(s.n for s in self.global_stats.values())+1
        bonus=0.15*math.sqrt(math.log(total_n+1)/(g.n+1))
        return base+bonus
    sem._strategy_value=types.MethodType(fast_only_value,sem)


def collect_phase(sem,env,coach,mode,n,max_frames,fc):
    coach.set_mode(mode); start=sem.shots_learned; rows=[]
    while sem.shots_learned-start<n and fc<max_frames:
        before=sem.shots_learned; obs=env.observe(); sv=sem.act(obs); cv=coach.act(obs)
        obs,evs=env.step(cv,sv,1/env.cfg.fps); sem.on_events(obs,evs); fc+=1
        if sem.shots_learned>before:
            ep=dict(sem.strategy_episodes[-1]); ctx=tuple(ep['ctx']); vals={a:sem._strategy_value(ctx,a) for a in sem.CONTACT_ACTIONS}; pref=max(vals,key=vals.get)
            rows.append({'intended_contact':float(ep['intended_contact']),'actual_contact':float(ep['actual_contact']),'reward':float(ep['reward']),'outcome':ep['outcome'],'preferred_action':float(pref)})
    return rows,fc


def frac(rows,target_sign,key='intended_contact'):
    return mean(sw.sign_of(r[key])==target_sign for r in rows) if rows else 0.0


def run_one(cp,seed,train_n,shock_n,recovery_n,mode,max_frames=900000):
    sem=AdaptiveSEMAgent.load_checkpoint(cp,seed=seed,reset_strategy=True); sem.set_intent_mode('strategy'); install_mode(sem,mode)
    env=PongEnv(PongConfig(seed=seed^0xA55A,win_score=999)); coach=sw.NonStationaryCoach(seed^0xC0A7,mode='weak_up'); fc=0
    train,fc=collect_phase(sem,env,coach,'weak_up',train_n,max_frames,fc)
    shock,fc=collect_phase(sem,env,coach,'weak_down',shock_n,max_frames,fc)
    recovery,fc=collect_phase(sem,env,coach,'weak_up',recovery_n,max_frames,fc)
    up=sw.correct_sign('weak_up'); down=sw.correct_sign('weak_down')
    first20=recovery[:min(20,len(recovery))]; last20=recovery[max(0,len(recovery)-20):]
    return {
      'seed':seed,'mode':mode,
      'train_last20_up_intent':frac(train[-20:],up),
      'shock_down_intent_fraction':frac(shock,down),
      'shock_up_retention_fraction':frac(shock,up),
      'recovery_first20_up_intent_fraction':frac(first20,up),
      'recovery_last20_up_intent_fraction':frac(last20,up),
      'recovery_mean_reward':mean(r['reward'] for r in recovery),
      'shock_mean_reward':mean(r['reward'] for r in shock),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=8); p.add_argument('--train',type=int,default=100); p.add_argument('--shock',type=int,default=10); p.add_argument('--recovery',type=int,default=50); p.add_argument('--seed',type=int,default=20260913); p.add_argument('--report',default='results/slow_stability_ablation.json'); args=p.parse_args()
    cp=ROOT/'checkpoints'/'sem_40runs.json.gz'; modes={'full':[],'fast_only':[]}
    for i in range(args.seeds):
        seed=args.seed+i*1009
        for m in modes: modes[m].append(run_one(cp,seed,args.train,args.shock,args.recovery,m))
        f=modes['full'][-1]; g=modes['fast_only'][-1]
        print(seed,'shock flip',round(f['shock_down_intent_fraction'],2),round(g['shock_down_intent_fraction'],2),'recovery20',round(f['recovery_first20_up_intent_fraction'],2),round(g['recovery_first20_up_intent_fraction'],2))
    agg={}
    metrics=['train_last20_up_intent','shock_down_intent_fraction','shock_up_retention_fraction','recovery_first20_up_intent_fraction','recovery_last20_up_intent_fraction','recovery_mean_reward','shock_mean_reward']
    for m,rs in modes.items(): agg[m]={k:mean(r[k] for r in rs) for k in metrics}
    paired={
      'shock_flip_reduction_full_vs_fast_only': mean(g['shock_down_intent_fraction']-f['shock_down_intent_fraction'] for f,g in zip(modes['full'],modes['fast_only'])),
      'recovery_first20_advantage_full_minus_fast_only': mean(f['recovery_first20_up_intent_fraction']-g['recovery_first20_up_intent_fraction'] for f,g in zip(modes['full'],modes['fast_only'])),
      'recovery_last20_advantage_full_minus_fast_only': mean(f['recovery_last20_up_intent_fraction']-g['recovery_last20_up_intent_fraction'] for f,g in zip(modes['full'],modes['fast_only'])),
      'full_better_recovery_first20_count':sum(f['recovery_first20_up_intent_fraction']>g['recovery_first20_up_intent_fraction'] for f,g in zip(modes['full'],modes['fast_only'])),
      'fast_only_better_recovery_first20_count':sum(f['recovery_first20_up_intent_fraction']<g['recovery_first20_up_intent_fraction'] for f,g in zip(modes['full'],modes['fast_only'])),
    }
    report={'status':'SEM_SLOW_MEMORY_STABILITY_ABLATION_COMPLETE','protocol':{'seeds':args.seeds,'train_A':args.train,'temporary_B_shock':args.shock,'recovery_A':args.recovery,'switch_signal_exposed':False,'memory_reset':False,'full':'fast+slow','fast_only':'slow ignored in action choice'},'aggregate':agg,'paired':paired,'runs':modes}
    out=ROOT/args.report; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print('report',out)
if __name__=='__main__': main()
