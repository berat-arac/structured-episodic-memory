import math, random
from dataclasses import dataclass

@dataclass
class PongConfig:
    width:int=1100
    height:int=650
    paddle_w:int=16
    paddle_h:int=110
    margin:int=30
    ball_r:int=9
    human_max_speed:float=1050.0
    sem_max_speed:float=1050.0
    ball_base_speed:float=330.0
    hit_accel:float=1.055
    max_ball_speed:float=1100.0
    bounce_angle_scale:float=1.02
    fps:int=60
    win_score:int=11
    seed:int=11

class PongEnv:
    def __init__(self,cfg=None):
        self.cfg=cfg or PongConfig()
        self.rng=random.Random(self.cfg.seed)
        self.reset_match()

    def reset_match(self):
        self.left_score=self.right_score=0
        self.rally_hits=0
        self.total_hits=0
        self.t=0.0
        self.left_y=self.right_y=self.cfg.height/2
        self._serve(self.rng.choice([-1,1]))

    def reset_round(self, direction=None):
        self.rally_hits=0
        self.left_y=self.right_y=self.cfg.height/2
        self._serve(direction or self.rng.choice([-1,1]))

    def _serve(self,dirx):
        c=self.cfg
        self.ball_x=c.width/2
        self.ball_y=c.height/2+self.rng.uniform(-65,65)
        ang=self.rng.uniform(-0.42,0.42)
        self.ball_vx=dirx*c.ball_base_speed*math.cos(ang)
        self.ball_vy=c.ball_base_speed*math.sin(ang)

    def observe(self):
        c=self.cfg
        return dict(t=self.t,width=c.width,height=c.height,paddle_h=c.paddle_h,paddle_w=c.paddle_w,
                    margin=c.margin,ball_r=c.ball_r,sem_max_speed=c.sem_max_speed,left_y=self.left_y,right_y=self.right_y,
                    ball_x=self.ball_x,ball_y=self.ball_y,ball_vx=self.ball_vx,ball_vy=self.ball_vy,
                    ball_speed=math.hypot(self.ball_vx,self.ball_vy),left_score=self.left_score,
                    right_score=self.right_score,rally_hits=self.rally_hits,total_hits=self.total_hits)

    def _move(self,y,velocity,max_speed,dt):
        v=max(-max_speed,min(max_speed,float(velocity)))
        y += v*dt
        return max(self.cfg.paddle_h/2,min(self.cfg.height-self.cfg.paddle_h/2,y))

    def step(self,left_velocity,right_velocity,dt=None):
        c=self.cfg; dt=dt or 1/c.fps
        self.t+=dt
        self.left_y=self._move(self.left_y,left_velocity,c.human_max_speed,dt)
        self.right_y=self._move(self.right_y,right_velocity,c.sem_max_speed,dt)
        speed_before=math.hypot(self.ball_vx,self.ball_vy)
        self.ball_x += self.ball_vx*dt
        self.ball_y += self.ball_vy*dt
        events=[]

        if self.ball_y-c.ball_r<=0:
            self.ball_y=c.ball_r; self.ball_vy=abs(self.ball_vy)
            events.append({'type':'wall','wall':'top'})
        elif self.ball_y+c.ball_r>=c.height:
            self.ball_y=c.height-c.ball_r; self.ball_vy=-abs(self.ball_vy)
            events.append({'type':'wall','wall':'bottom'})

        lx=c.margin+c.paddle_w/2
        rx=c.width-c.margin-c.paddle_w/2
        if self.ball_vx<0 and self.ball_x-c.ball_r<=lx+c.paddle_w/2:
            if abs(self.ball_y-self.left_y)<=c.paddle_h/2:
                contact=(self.ball_y-self.left_y)/(c.paddle_h/2)
                self.ball_x=lx+c.paddle_w/2+c.ball_r
                sp=min(c.max_ball_speed,math.hypot(self.ball_vx,self.ball_vy)*c.hit_accel)
                ang=contact*c.bounce_angle_scale
                self.ball_vx=abs(sp*math.cos(ang)); self.ball_vy=sp*math.sin(ang)
                self.rally_hits+=1; self.total_hits+=1
                events.append({'type':'hit','side':'left','contact':contact,'speed_before':speed_before,'speed_after':sp})
        if self.ball_vx>0 and self.ball_x+c.ball_r>=rx-c.paddle_w/2:
            if abs(self.ball_y-self.right_y)<=c.paddle_h/2:
                contact=(self.ball_y-self.right_y)/(c.paddle_h/2)
                self.ball_x=rx-c.paddle_w/2-c.ball_r
                sp=min(c.max_ball_speed,math.hypot(self.ball_vx,self.ball_vy)*c.hit_accel)
                ang=math.pi-contact*c.bounce_angle_scale
                self.ball_vx=sp*math.cos(ang); self.ball_vy=sp*math.sin(ang)
                self.rally_hits+=1; self.total_hits+=1
                events.append({'type':'hit','side':'right','contact':contact,'speed_before':speed_before,'speed_after':sp})

        if self.ball_x<-45:
            self.right_score+=1
            events.append({'type':'score','side':'right','rally_hits':self.rally_hits})
            self.reset_round(direction=-1)
        elif self.ball_x>c.width+45:
            self.left_score+=1
            events.append({'type':'score','side':'left','rally_hits':self.rally_hits})
            self.reset_round(direction=1)

        return self.observe(),events
