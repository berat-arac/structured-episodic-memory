from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame

from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def load_sem(checkpoint: Path, seed: int, fresh_opponent: bool):
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            "Run `python train_sem.py` first."
        )
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=fresh_opponent)
    sem.set_intent_mode("strategy")
    sem.set_eval_mode(False)
    return sem


def main(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    sem = load_sem(checkpoint, args.seed, fresh_opponent=not args.keep_trained_strategy)

    cfg = PongConfig(seed=args.game_seed, win_score=999)
    env = PongEnv(cfg)

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Project B — Human vs Trained Oracle-Free SEM")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 20)
    small = pygame.font.SysFont("Consolas", 15)
    running = True
    debug = True

    baseline_motor_outcomes = sem.motor_outcomes
    baseline_motor_hits = sem.motor_hits
    baseline_motor_misses = sem.motor_misses

    def draw_text(s, pos, color=(220, 230, 240), f=None):
        screen.blit((f or font).render(s, True, color), pos)

    while running:
        dt = min(.035, clock.tick(cfg.fps) / 1000.0)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    # New scoreboard/match, all SEM learning is preserved.
                    env.reset_match()
                elif e.key == pygame.K_o:
                    # Forget only the current opponent model, keep motor skill.
                    sem.reset_opponent_memory()
                elif e.key == pygame.K_x:
                    # Restore the pretrained checkpoint and start a fresh human model.
                    sem = load_sem(checkpoint, args.seed, fresh_opponent=True)
                    env.reset_match()
                    baseline_motor_outcomes = sem.motor_outcomes
                    baseline_motor_hits = sem.motor_hits
                    baseline_motor_misses = sem.motor_misses
                elif e.key == pygame.K_n:
                    env.reset_round()
                elif e.key == pygame.K_d:
                    debug = not debug

        keys = pygame.key.get_pressed()
        human_v = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            human_v -= cfg.human_max_speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            human_v += cfg.human_max_speed

        obs = env.observe()
        sem_v = sem.act(obs)
        obs, events = env.step(human_v, sem_v, dt=dt)
        sem.on_events(obs, events)

        screen.fill((13, 16, 22))
        for y in range(0, cfg.height, 22):
            pygame.draw.rect(screen, (54, 61, 73), (cfg.width // 2 - 2, y, 4, 12))
        pygame.draw.rect(screen, (116, 219, 255), (cfg.margin, int(env.left_y - cfg.paddle_h / 2), cfg.paddle_w, cfg.paddle_h), border_radius=5)
        pygame.draw.rect(screen, (255, 133, 83), (cfg.width - cfg.margin - cfg.paddle_w, int(env.right_y - cfg.paddle_h / 2), cfg.paddle_w, cfg.paddle_h), border_radius=5)
        pygame.draw.circle(screen, (245, 247, 250), (int(env.ball_x), int(env.ball_y)), cfg.ball_r)

        draw_text(f"YOU {env.left_score}", (25, 18), (135, 225, 255))
        draw_text(f"SEM {env.right_score}", (cfg.width - 145, 18), (255, 175, 120))
        draw_text(f"ball {obs['ball_speed']:6.1f}   rally hits {obs['rally_hits']}", (25, 48), (190, 205, 220), small)
        draw_text(
            "W/S move   R new match/keep learning   O forget opponent   X reload pretrained   N round   D debug   ESC save+quit",
            (25, cfg.height - 28), (150, 170, 190), small,
        )

        if debug:
            sm = sem.summary()
            session_out = sem.motor_outcomes - baseline_motor_outcomes
            session_hits = sem.motor_hits - baseline_motor_hits
            session_misses = sem.motor_misses - baseline_motor_misses
            x, y = cfg.width - 430, 62
            pygame.draw.rect(screen, (20, 25, 34), (x - 14, y - 10, 414, 430), border_radius=8)
            draw_text("TRAINED ORACLE-FREE SEM", (x, y), (255, 195, 145), small); y += 24
            draw_text(f"pretrained outcomes : {baseline_motor_outcomes}", (x, y), f=small); y += 19
            draw_text(f"human-session motor : {session_out}  hit {session_hits} / miss {session_misses}", (x, y), f=small); y += 19
            draw_text(f"motor explore       : {sm['motor_explore']:.3f}", (x, y), f=small); y += 19
            draw_text(f"motor command       : {sm['last_motor_action']:+.2f}", (x, y), f=small); y += 19
            draw_text(f"smooth velocity     : {sm['applied_velocity']:+7.1f}", (x, y), f=small); y += 19
            last_miss = sm.get("last_miss_audit")
            if last_miss:
                draw_text(f"last miss cause     : {last_miss['likely_cause']}", (x, y), (255, 170, 155), small); y += 19
                draw_text(f"  decisions/explore : {last_miss['num_decisions']}/{last_miss['num_exploratory_decisions']}  vmax {last_miss['max_ball_speed']:.0f}", (x, y), (190, 200, 215), small); y += 20
            else:
                draw_text("last miss cause     : none this process", (x, y), (175, 205, 180), small); y += 20

            draw_text("LIVE TACTIC MODEL (THIS OPPONENT)", (x, y), (175, 220, 255), small); y += 21
            draw_text(f"shots learned       : {sm['shots_learned']}", (x, y), f=small); y += 19
            draw_text(f"strategy explore    : {sm['strategy_explore']:.3f}", (x, y), f=small); y += 19
            draw_text(f"current intent      : {sm['current_contact_intent']:+.2f}", (x, y), (255, 215, 150), small); y += 19
            draw_text(f"last actual contact : {sm['last_contact']:+.2f}", (x, y), f=small); y += 19
            draw_text(f"best observed zone  : {sm['best_observed_contact']:+.2f}", (x, y), (180, 240, 185), small); y += 19
            draw_text(f"mean intent error   : {sm['mean_intent_error']:.3f}", (x, y), f=small); y += 23
            draw_text("zone        fast   slow    n", (x, y), (170, 190, 210), small); y += 18
            for a in sem.CONTACT_ACTIONS:
                st = sem.global_stats[a]
                marker = ">" if abs(a - sm["best_observed_contact"]) < 1e-9 and st.n else " "
                draw_text(f"{marker}{a:+.2f}       {st.fast:5.2f}  {st.slow:5.2f}  {st.n:3d}", (x, y), f=small)
                y += 17

        pygame.display.flip()

    out_ckpt = ROOT / "checkpoints" / "sem_after_human.json.gz"
    sem.save_checkpoint(out_ckpt, include_strategy=True, metadata={"source_checkpoint": str(checkpoint)})
    summary = {
        "final_score": {"human": env.left_score, "sem": env.right_score},
        "source_checkpoint": str(checkpoint),
        "adapted_checkpoint": str(out_ckpt),
        "sem": sem.summary(),
    }
    out_json = ROOT / "results" / "last_human_match.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit_json = ROOT / "results" / "last_human_miss_audit.json"
    audit_json.write_text(json.dumps({
        "final_score": summary["final_score"],
        "source_checkpoint": str(checkpoint),
        "misses": list(sem.miss_audits),
    }, indent=2), encoding="utf-8")
    pygame.quit()
    print("saved", out_json)
    print("saved", audit_json)
    print("saved", out_ckpt)


def parse_args():
    p = argparse.ArgumentParser(description="Play against a pretrained oracle-free SEM.")
    p.add_argument("--checkpoint", default="checkpoints/sem_40runs.json.gz")
    p.add_argument("--seed", type=int, default=77001, help="SEM tie-break/exploration seed")
    p.add_argument("--game-seed", type=int, default=11)
    p.add_argument("--keep-trained-strategy", action="store_true", help="keep coach strategy memory instead of learning you from scratch")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
