import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import math
import sys

pygame.init()

# === SETTING LAYAR ===
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rocket Landing Simulation")

# === WARNA ===
BLACK = (10, 10, 30)
BLUE = (0, 150, 255)
ORANGE = (255, 140, 0)
YELLOW = (255, 200, 0)
RED = (255, 60, 0)
WHITE = (255, 255, 255)

# === GAMBAR ===
rocket_img = pygame.image.load('rocket.png').convert_alpha()
platform_img = pygame.image.load("target.png").convert_alpha()

# === SKALA GAMBAR ===
rocket_img = pygame.transform.scale(rocket_img, (70, 100))
platform_img = pygame.transform.scale(platform_img, (160, 40))

# === POSISI AWAL ===
t = 0
rocket_x, rocket_y = 100, HEIGHT - 100
platform_x = WIDTH - 300
platform_y = HEIGHT - 60
platform_speed = 7
clock = pygame.time.Clock()
landed = False

# === REWARD SYSTEM ===
reward = 0
best_reward = 0

# === FUNGSI LINTASAN ===
def rocket_path(t):
    """
    Trajektori roket:
    - Naik cukup tinggi (~300px dari bawah)
    - Tidak terlalu jauh ke kanan
    - Mendarat di sekitar tengah layar
    """
    x = 100 + 1.3 * t             # gerakan horizontal lebih lambat
    y = HEIGHT - 100 - (4.8 * t) + 0.012 * (t ** 2)  # parabolanya lebih curam
    return x, y

# === API ROKET ===
def draw_fire(x, y):
    for i in range(8):
        flame_x = x + 35 + random.randint(-6, 6)
        flame_y = y + 95 + random.randint(-2, 10)
        color = random.choice([ORANGE, YELLOW, RED])
        radius = random.randint(3, 6)
        pygame.draw.circle(screen, color, (int(flame_x), int(flame_y)), radius)

# === REWARD ===
def calculate_reward(rocket_rect, platform_rect, landed):
    dx = abs(rocket_rect.centerx - platform_rect.centerx)
    dy = abs(rocket_rect.centery - platform_rect.centery)
    distance = math.sqrt(dx ** 2 + dy ** 2)
    r = 0
    if landed:
        r += 100
    else:
        r += max(0, 50 - distance * 0.05)
        if rocket_rect.bottom > HEIGHT:
            r -= 50
    return r

# === GAME LOOP ===
running = True
while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and platform_x > 0:
        platform_x -= platform_speed
    if keys[pygame.K_RIGHT] and platform_x < WIDTH - 160:
        platform_x += platform_speed

    # === GERAK ROKET ===
    if not landed:
        rocket_x, rocket_y = rocket_path(t)
        t += 1
        angle = -math.sin(t * 0.015) * 20
        rotated_rocket = pygame.transform.rotate(rocket_img, angle)
        rocket_rect = rotated_rocket.get_rect(center=(rocket_x + 35, rocket_y + 50))

        draw_fire(rocket_x, rocket_y)

        # Reset jika keluar layar
        if rocket_y > HEIGHT or rocket_x > WIDTH:
            t = 0
            rocket_x, rocket_y = 100, HEIGHT - 100
            reward -= 50

        # Deteksi tabrakan dengan platform
        platform_rect = pygame.Rect(platform_x, platform_y, 160, 40)
        if rocket_rect.colliderect(platform_rect):
            landed = True
            reward += 100

    else:
        rocket_rect = rocket_img.get_rect(center=(platform_x + 80, platform_y - 40))
        platform_rect = pygame.Rect(platform_x, platform_y, 160, 40)

    # === REWARD FRAME ===
    frame_reward = calculate_reward(rocket_rect, platform_rect, landed)
    reward += frame_reward
    best_reward = max(best_reward, reward)

    # === GAMBAR OBJEK ===
    screen.blit(platform_img, (platform_x, platform_y))
    screen.blit(rotated_rocket if not landed else rocket_img, rocket_rect)

    # === TEKS INFORMASI ===
    font = pygame.font.SysFont(None, 30)
    text1 = font.render(f"Reward: {reward:.1f}", True, WHITE)
    text2 = font.render(f"Best: {best_reward:.1f}", True, YELLOW)
    screen.blit(text1, (20, 20))
    screen.blit(text2, (20, 50))

    if landed:
        font_big = pygame.font.SysFont(None, 48)
        msg = font_big.render("Landed Successfully!", True, BLUE)
        screen.blit(msg, (WIDTH // 2 - 200, HEIGHT // 2 - 50))
        

    pygame.display.flip()
    clock.tick(60)

# === ENVIRONMENT UNTUK RL ===
class SimpleRocketEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode

        # === Physics ===
        self.dt, self.m, self.g = 0.025, 3.0, 9.81
        self.F_main = 600.0  # thrust utama (vertikal)
        self.F_side = 150.0  # thrust lateral kecil untuk arah kanan
        self.b_linear = 0.1
        self.b_angular = 0.05

        # === Screen ===
        self.screen_w, self.screen_h = 960, 480
        self.floor_y = 10.0

        # === Rocket ===
        self.w, self.h = 30.0, 60.0
        self.I = (1/12) * self.m * (self.w*2 + self.h*2)

        # === Target (platform biru) ===
        self.target_pos = np.array([750.0, 40.0], np.float32)
        self.target_w, self.target_h = 200.0, 50.0
        self.target_vx = 50.0
        self.target_min_x = 500.0
        self.target_max_x = 900.0

        # === Launch Pad ===
        self.launch_pad_pos = np.array([100.0, 50.0], dtype=np.float32)
        self.pad_w, self.pad_h = 40.0, 40.0

        # === Observation & Actions ===
        high = np.array([
            self.screen_w, self.screen_h,
            np.finfo(np.float32).max, np.finfo(np.float32).max,
            1., 1.,
            np.finfo(np.float32).max,
            self.screen_w, self.screen_h
        ], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # === Rendering ===
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Rocket Arc Simulation (Upward Curve)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
            base = os.path.dirname(__file__)
            load = lambda fn: pygame.image.load(os.path.join(base, fn)).convert_alpha()
            self.rocket_img = pygame.transform.scale(load("rocket.png"), (int(self.w), int(self.h)))
            self.target_img = pygame.transform.scale(load("target.png"),
                                                    (int(self.target_w), int(self.target_h)))

        self.state = np.zeros(8, dtype=np.float32)
        self.last_action = 0
        self.step_count = 0
        self.max_steps = 800

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([100., 120., 20., 60., 0., 1., 0., 0., 0.], np.float32)
        self.last_action = 0
        self.step_count = 0
        return self.state, {}

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
        self.step_count += 1
        x, y, vx, vy, sinθ, cosθ, ω, dx, dy = self.state
        Fx = Fy = 0.0
        θ = 0.0

        if action == 1:
            Fy += self.F_main
        elif action == 2:
            Fx += self.F_side

        Fx += -self.b_linear * vx
        Fy += -self.b_linear * vy
        Fy -= self.m * self.g

        vx += (Fx / self.m) * self.dt
        vy += (Fy / self.m) * self.dt
        x += vx * self.dt
        y += vy * self.dt

        x = np.clip(x, 0, self.screen_w)
        y = np.clip(y, 0, self.screen_h)

        tx, ty = self.target_pos
        dx, dy = x - tx, y - ty
        dist = np.hypot(dx, dy)
        speed = np.hypot(vx, vy)
        angle_penalty = abs(θ)

        reward = -0.03 * abs(dx) - 0.05 * abs(dy) - 0.02 * speed - 0.1 * angle_penalty
        reward += 1.0 * cosθ

        if (tx - self.target_w/2 <= x <= tx + self.target_w/2) and (abs(y - ty) < 20) and (abs(vy) < 5):
            reward += 300
            terminated = True
        elif y <= self.floor_y:
            reward -= 100
            terminated = True
        else:
            terminated = False

        truncated = self.step_count >= self.max_steps

        # Pergerakan target (kanan–kiri)
        tx += self.target_vx * self.dt
        if tx < self.target_min_x or tx > self.target_max_x:
            self.target_vx = -self.target_vx
        tx = np.clip(tx, self.target_min_x, self.target_max_x)
        self.target_pos[0] = tx

        self.state = np.array([x, y, vx, vy, 0., 1., 0., dx, dy], np.float32)
        self.last_action = action
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode != 'human':
            return
        x, y, vx, vy, sinθ, cosθ, ω, dx, dy = self.state
        self.screen.fill((0, 0, 20))
        y_screen = self.screen_h - y
        rocket = pygame.transform.rotate(self.rocket_img, 0)
        rect = rocket.get_rect(center=(x, y_screen))
        self.screen.blit(rocket, rect.topleft)
        tx, ty = self.target_pos
        tx_draw = tx - self.target_w / 2
        ty_draw = self.screen_h - ty - self.target_h / 2
        self.screen.blit(self.target_img, (int(tx_draw), int(ty_draw)))
        info = f"x={x:.1f} y={y:.1f} vx={vx:.1f} vy={vy:.1f}"
        self.screen.blit(self.font.render(info, True, (255, 255, 255)), (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
