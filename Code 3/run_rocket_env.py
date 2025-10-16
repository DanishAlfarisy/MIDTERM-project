import pygame
import numpy as np
from stable_baselines3 import DQN
from rocket_env import SimpleRocketEnv  # pastikan file rocket_env.py ada di folder yang sama

# ------------------------------------------------
# Fungsi simulasi rollout (prediksi lintasan)
# ------------------------------------------------
def simulate_rollout(model, state, n_steps=400, deterministic=False):
    """
    Simulasikan beberapa langkah ke depan dari state saat ini menggunakan model.predict().
    Digunakan untuk memprediksi ke mana roket akan jatuh.
    """
    env_sim = SimpleRocketEnv()  # environment dummy untuk simulasi
    env_sim.set_state(state)

    traj = []
    obs = env_sim._get_obs()
    terminated, truncated = False, False

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()  # <-- ini yang penting
        done = terminated or truncated
        traj.append((env_sim.state[0], env_sim.state[1]))
        if terminated or truncated:
            break

    return env_sim.state[0], env_sim.state[1], traj


# ------------------------------------------------
# Konversi koordinat dunia ke layar pygame
# ------------------------------------------------
def world_to_screen(x, y, screen_w, screen_h, scale=20):
    sx = int(screen_w // 2 + x * scale)
    sy = int(screen_h - y * scale)
    return sx, sy


# ------------------------------------------------
# Program utama
# ------------------------------------------------
if __name__ == "__main__":
    # Load model DQN hasil training
    model = DQN.load("./models/dqn_baseline.zip")

    # Buat environment roket
    env = SimpleRocketEnv(render_mode="human")
    obs, _ = env.reset()
    done = False

    # Setup pygame
    pygame.init()
    screen_w, screen_h = 960, 540
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Rocket Landing Prediction (AI)")

    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Ambil state sekarang dari env
        state = env._get_obs()

        # ---- Prediksi lintasan masa depan ----
        preds = []
        for _ in range(5):  # 5 simulasi monte carlo
            fx, fy, traj = simulate_rollout(model, state, n_steps=400, deterministic=False)
            preds.append((fx, fy))

        # Rata-rata titik jatuh
        avg_x = np.mean([p[0] for p in preds])
        avg_y = np.mean([p[1] for p in preds])

        # ---- Jalankan langkah model di env utama ----
        action, _ = model.predict(state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # ---- Render visual ----
        screen.fill((0, 0, 0))  # background hitam

        # Platform biru (target)
        platform_x = env.target_pos[0]
        platform_y = env.target_pos[1]
        px1, py1 = world_to_screen(platform_x - 3.0, platform_y, screen_w, screen_h)
        px2, py2 = world_to_screen(platform_x + 3.0, platform_y + 0.5, screen_w, screen_h)
        pygame.draw.rect(screen, (0, 100, 255), pygame.Rect(px1, py2, px2 - px1, py1 - py2))

        # Roket hijau (posisi saat ini)
        rx, ry = world_to_screen(env.state[0], env.state[1], screen_w, screen_h)
        pygame.draw.circle(screen, (0, 255, 0), (rx, ry), 6)

        # Titik prediksi lintasan merah
        for fx, fy in preds:
            sx, sy = world_to_screen(fx, fy, screen_w, screen_h)
            pygame.draw.circle(screen, (255, 0, 0), (sx, sy), 5)

        # Titik rata-rata prediksi kuning
        ax, ay = world_to_screen(avg_x, avg_y, screen_w, screen_h)
        pygame.draw.circle(screen, (255, 255, 0), (ax, ay), 7)

        # Update tampilan
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()