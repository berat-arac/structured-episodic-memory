from __future__ import annotations
import argparse, copy, json, types
from pathlib import Path
from statistics import mean
from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT=Path(__file__).resolve().parent

def install(sem, mode):
    sem.aim_weight=0.0
    if mode=='full': return
    if mode not in ('fast_only','slow_only'): raise ValueError(mode)
    def estimate(self, keys, action):
        weighted=0.; ws=0.; evidence=0; disagreement=0.
        for level,key in enumerate(keys):
            st=self.table[key][action]; specificity=1/(1+0.20*level); confidence=min(1.,st.n/7.); w=specificity*(0.25+0.75*confidence)
            value=st.fast if mode=='fast_only' else st.slow
            weighted += w*value; ws += w; evidence += st.n; disagreement=max(disagreement,abs(st.fast-st.slow))
        return weighted/max(ws,1e-9), evidence, disagreement
    sem.motor._estimate=types.MethodType(estimate,sem.motor)

def lat(seq,win=20,thr=.75):
    for end in range(win,len(seq)+1):
        if sum(seq[end-win:end])/win>=thr:return end
    return None

def phase(sem,seed,idx,n,invert):
    env=PongEnv(PongConfig(seed=seed,win_score=999)); coach=build_coach(seed^0x2A77,900+idx); seq=[]; f=0
    while len(seq)<n and f<400000:
        o=env.observe(); cmd=sem.act(o); app=-cmd if invert else cmd
        o,ev=env.step(coach.act(o),app,1/env.cfg.fps); sem.on_events(o,ev)
        for e in ev:
            if e['type']=='hit' and e['side']=='right': seq.append(1)
            elif e['type']=='score' and e['side']=='left': seq.append(0)
        f+=1
    return {'early20':sum(seq[:20])/20,'first50':sum(seq[:50])/50,'first100':sum(seq[:100])/100,'late20':sum(seq[-20:])/20,'rate':sum(seq)/len(seq),'lat75':lat(seq),'sequence':seq}

def run(args):
    cp=Path(args.checkpoint); cp=cp if cp.is_absolute() else ROOT/cp
    rows=[]
    for i in range(args.seeds):
        seed=args.base_seed+i*317
        base=AdaptiveSEMAgent.load_checkpoint(cp,seed=seed^0x7731,reset_strategy=True); base.set_intent_mode('neutral'); base.set_eval_mode(False); base.aim_weight=0.0
        a1=phase(base,seed,i,args.a1,False); b1=phase(base,seed,i,args.b,True); a2=phase(base,seed,i,args.a2,False)
        branches={}
        for mode in ['full','fast_only','slow_only']:
            s=copy.deepcopy(base); install(s,mode); branches[mode]=phase(s,seed,i,args.b,True)
        rows.append({'seed':seed,'history':{'A1':a1,'B1':b1,'A2':a2},'B2_branches':branches})
        print(seed, {m:(branches[m]['lat75'],branches[m]['first100']) for m in branches})
    agg={}
    for m in ['full','fast_only','slow_only']:
        rs=[r['B2_branches'][m] for r in rows]; l=[x['lat75'] for x in rs if x['lat75'] is not None]
        agg[m]={'mean_first50':mean(x['first50'] for x in rs),'mean_first100':mean(x['first100'] for x in rs),'mean_late20':mean(x['late20'] for x in rs),
                'mean_rate':mean(x['rate'] for x in rs),'adapted':len(l),'mean_lat75':mean(l) if l else None,'lat75_values':[x['lat75'] for x in rs]}
    paired=[]
    for r in rows:
        b=r['B2_branches']; paired.append({'seed':r['seed'],'full_vs_fast_first100':b['full']['first100']-b['fast_only']['first100'],
          'full_vs_slow_first100':b['full']['first100']-b['slow_only']['first100'],'full_lat':b['full']['lat75'],'fast_lat':b['fast_only']['lat75'],'slow_lat':b['slow_only']['lat75']})
    report={'status':'MOTOR_RETENTION_BRANCH_PROBE_COMPLETE','protocol':{'shared_history':'A1 normal -> B1 inverted -> A2 normal, then deepcopy','B2_branches':['full','fast_only','slow_only'],'identical_prebranch_memory':True,'aim_weight':0.0},'aggregate':agg,'paired':paired,'rows':rows}
    out=Path(args.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':agg,'paired':paired},indent=2)); print('report',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/sem_40runs.json.gz'); p.add_argument('--base-seed',type=int,default=20270601); p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--a1',type=int,default=100); p.add_argument('--b',type=int,default=220); p.add_argument('--a2',type=int,default=220); p.add_argument('--output',default='results/motor_retention_branch_probe.json'); run(p.parse_args())
