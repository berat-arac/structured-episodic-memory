from __future__ import annotations
import argparse, json, math, random, types
from pathlib import Path
from statistics import mean, median
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent


def install_fast_only(sem):
    def fast_only_value(self, ctx, action):
        g=self.global_stats[action]; c=self.ctx_stats[ctx][action]
        cw=min(0.72,c.n/7.0)
        base=(1-cw)*g.fast+cw*c.fast
        total_n=sum(s.n for s in self.global_stats.values())+1
        bonus=0.15*math.sqrt(math.log(total_n+1)/(g.n+1))
        return base+bonus
    sem._strategy_value=types.MethodType(fast_only_value,sem)


def make_obs():
    return {'left_y':325.0,'height':650.0,'ball_speed':520.0}


def one(seed,steps,mode,p_target,p_other):
    sem=AdaptiveSEMAgent(seed=seed); sem.reset_opponent_memory(); sem.opp_vy=0.0
    if mode=='fast_only': install_fast_only(sem)
    rng=random.Random(seed^0x9911); obs=make_obs(); ctx=sem.strategy_context(obs); target=0.44
    choices=[]; prefs=[]; rewards=[]
    for t in range(steps):
        choice,_=sem.preferred_contact(obs,explore=True)
        p=p_target if choice==target else p_other
        reward=1.0 if rng.random()<p else 0.0
        sem.global_stats[choice].update(reward); sem.ctx_stats[ctx][choice].update(reward)
        sem.shots_learned += 1
        sem.strategy_explore=max(0.080,0.42*math.exp(-sem.shots_learned/48.0))
        vals={a:sem._strategy_value(ctx,a) for a in sem.CONTACT_ACTIONS}; pref=max(vals,key=vals.get)
        choices.append(choice); prefs.append(pref); rewards.append(reward)
    tail=min(1000,steps); ch=choices[-tail:]; pr=prefs[-tail:]
    switches=sum(a!=b for a,b in zip(pr,pr[1:]))
    return {
      'seed':seed,'mode':mode,
      'target_choice_fraction_tail':sum(x==target for x in ch)/tail,
      'target_preferred_fraction_tail':sum(x==target for x in pr)/tail,
      'preferred_switches_per_1000_tail':switches*1000/max(1,tail-1),
      'mean_reward_tail':mean(rewards[-tail:]),
      'final_fast_target':sem.global_stats[target].fast,
      'final_slow_target':sem.global_stats[target].slow,
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=64); p.add_argument('--steps',type=int,default=4000); p.add_argument('--p-target',type=float,default=0.58); p.add_argument('--p-other',type=float,default=0.52); p.add_argument('--seed',type=int,default=20260913); p.add_argument('--report',default='results/slow_noise_isolation.json'); args=p.parse_args()
    modes={'full':[],'fast_only':[]}
    for i in range(args.seeds):
        seed=args.seed+i*97
        modes['full'].append(one(seed,args.steps,'full',args.p_target,args.p_other)); modes['fast_only'].append(one(seed,args.steps,'fast_only',args.p_target,args.p_other))
    metrics=['target_choice_fraction_tail','target_preferred_fraction_tail','preferred_switches_per_1000_tail','mean_reward_tail']
    agg={m:{k:mean(r[k] for r in rs) for k in metrics} for m,rs in modes.items()}
    paired={
      'target_choice_advantage_full_minus_fast':mean(f['target_choice_fraction_tail']-g['target_choice_fraction_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
      'target_preferred_advantage_full_minus_fast':mean(f['target_preferred_fraction_tail']-g['target_preferred_fraction_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
      'switch_reduction_full_vs_fast':mean(g['preferred_switches_per_1000_tail']-f['preferred_switches_per_1000_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
      'reward_advantage_full_minus_fast':mean(f['mean_reward_tail']-g['mean_reward_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
      'full_better_target_pref_seeds':sum(f['target_preferred_fraction_tail']>g['target_preferred_fraction_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
      'fast_better_target_pref_seeds':sum(f['target_preferred_fraction_tail']<g['target_preferred_fraction_tail'] for f,g in zip(modes['full'],modes['fast_only'])),
    }
    report={'status':'SEM_SLOW_MEMORY_NOISE_ISOLATION_COMPLETE','protocol':{'synthetic_strategy_isolation':True,'seeds':args.seeds,'steps':args.steps,'target_action':0.44,'target_success_probability':args.p_target,'other_success_probability':args.p_other,'fixed_context':True,'note':'Isolates strategy value memory from Pong motor/contact execution; same exploration and value updates as SEM strategy memory.'},'aggregate':agg,'paired':paired,'runs':modes}
    out=ROOT/args.report; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print(out)
if __name__=='__main__': main()
