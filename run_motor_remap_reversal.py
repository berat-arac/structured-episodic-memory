from __future__ import annotations
import argparse,json
from pathlib import Path
from coach_opponents import build_coach
from pong_core import PongConfig,PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent

def run_phase(sem,seed,run_index,n,invert):
    env=PongEnv(PongConfig(seed=seed,win_score=999))
    coach=build_coach(seed^0x2A77,900+run_index)
    out=[]; frames=0
    while len(out)<n and frames<300000:
        obs=env.observe(); command=sem.act(obs)
        applied=-command if invert else command
        obs,events=env.step(coach.act(obs),applied,1/env.cfg.fps)
        sem.on_events(obs,events)
        for e in events:
            if e['type']=='hit' and e['side']=='right': out.append(1)
            elif e['type']=='score' and e['side']=='left': out.append(0)
        frames+=1
    k=min(20,max(1,len(out)//4))
    return {'invert':invert,'outcomes':len(out),'hit_rate':sum(out)/max(1,len(out)),
            'early_hit_rate':sum(out[:k])/max(1,k),'late_hit_rate':sum(out[-k:])/max(1,k),
            'early_misses':k-sum(out[:k]),'late_misses':k-sum(out[-k:]),'sequence':out}

def run(args):
    ckpt=Path(args.checkpoint); ckpt=ckpt if ckpt.is_absolute() else ROOT/ckpt
    rows=[]
    for i in range(args.seeds):
        base=args.base_seed+i*317
        sem=AdaptiveSEMAgent.load_checkpoint(ckpt,seed=base^0x7731,reset_strategy=True)
        sem.set_intent_mode('neutral'); sem.set_eval_mode(False)
        a1=run_phase(sem,base,i,args.a_outcomes,False)
        b=run_phase(sem,base,i,args.b_outcomes,True)
        a2=run_phase(sem,base,i,args.a_outcomes,False)
        rows.append({'seed':base,'A1_normal':a1,'B_inverted':b,'A2_normal_return':a2})
        print(f"seed {base}: A1 {a1['early_hit_rate']:.2f}->{a1['late_hit_rate']:.2f} | "
              f"B {b['early_hit_rate']:.2f}->{b['late_hit_rate']:.2f} | "
              f"A2 {a2['early_hit_rate']:.2f}->{a2['late_hit_rate']:.2f}")
    def m(path,key): return sum(r[path][key] for r in rows)/len(rows)
    summary={
      'A1_early':m('A1_normal','early_hit_rate'),'A1_late':m('A1_normal','late_hit_rate'),
      'B_early':m('B_inverted','early_hit_rate'),'B_late':m('B_inverted','late_hit_rate'),
      'A2_early':m('A2_normal_return','early_hit_rate'),'A2_late':m('A2_normal_return','late_hit_rate'),
      'B_recovery_delta':m('B_inverted','late_hit_rate')-m('B_inverted','early_hit_rate'),
      'A2_recovery_delta':m('A2_normal_return','late_hit_rate')-m('A2_normal_return','early_hit_rate'),
    }
    payload={'status':'MOTOR_REMAP_REVERSAL_COMPLETE','seeds':args.seeds,'A_outcomes':args.a_outcomes,'B_outcomes':args.b_outcomes,
             'protocol':{'A1':'normal actuator','B':'SEM command sign inverted externally','A2':'normal restored',
                         'switch_signaled_to_sem':False,'memory_reset_between_phases':False},
             'summary':summary,'rows':rows}
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('report:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz'); p.add_argument('--base-seed',type=int,default=20270401)
    p.add_argument('--seeds',type=int,default=1); p.add_argument('--a-outcomes',type=int,default=100); p.add_argument('--b-outcomes',type=int,default=220); p.add_argument('--output',default='results/motor_remap_smoke.json')
    run(p.parse_args())
