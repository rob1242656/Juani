import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import multiprocessing
import torch.nn as nn
import math
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# --- CONSTANTES ---
COLL_SUELO = 1
COLL_OBSTACULO = 2
COLL_TORSO = 10
COLL_PIE_IZQ = 11
COLL_PIE_DER = 12
COLL_MUSLO = 13

# Config
ANCHO, ALTO = 1000, 600
NOMBRE_MODELO = "sonso_v91_smart_manager"

class SonsoEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(SonsoEnv, self).__init__()
        self.render_mode = render_mode
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # --- V9.1: 30 INPUTS (29 SENSORES + 1 ORDEN DEL MANAGER) ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        
        # Variables HRL (Manager)
        self.goal_velocity = 0.0 # La orden actual del jefe
        self.dist_lidar_lejos = 1.0 # Memoria del sensor para el Manager
        
        # Variables DR
        self.grav_y = 900.0
        self.friccion_dr = 1.5
        
        if self.render_mode == 'human':
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((ANCHO, ALTO))
            pygame.display.set_caption(f"Sonso IA: {NOMBRE_MODELO} (HRL)")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.font = pygame.font.SysFont("Consolas", 18)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = pymunk.Space()
        
        # --- DOMAIN RANDOMIZATION ---
        self.grav_y = np.random.uniform(700.0, 1100.0)
        self.space.gravity = (0.0, self.grav_y)
        self.friccion_dr = np.random.uniform(0.4, 2.0)
        masa_random = np.random.uniform(6.0, 10.0)
        
        # Matrix Registro
        partes_robot = [COLL_TORSO, COLL_PIE_IZQ, COLL_PIE_DER, COLL_MUSLO]
        entorno_types = [COLL_SUELO, COLL_OBSTACULO]

        for parte in partes_robot:
            for entorno in entorno_types:
                h = self.space.add_collision_handler(parte, entorno)
                h.begin = self._cb_begin
                h.separate = self._cb_separate
                h.post_solve = self._cb_post_solve 
        
        self.c_izq_counter = 0; self.c_der_counter = 0
        self.prev_contact_izq = False; self.prev_contact_der = False
        
        self.impacto_izq = 0.0; self.impacto_der = 0.0
        self.impacto_torso = 0.0; self.impacto_muslo = 0.0
        
        self.tiempo = 0
        self.prev_action = np.zeros(4)
        self.push_visual = 0
        self.lista_obstaculos = [] 
        
        # Reset HRL
        self.goal_velocity = 0.0
        self.dist_lidar_lejos = 1.0
        
        self._crear_escenario()
        self._crear_robot(200, 420, masa_random) 
        self.next_obstacle_x = 1000 
        
        return self._get_obs(), {}

    def _identificar_parte(self, arbiter):
        for shape in arbiter.shapes:
            ct = shape.collision_type
            if ct in [COLL_TORSO, COLL_PIE_IZQ, COLL_PIE_DER, COLL_MUSLO]: return ct
        return None

    def _cb_begin(self, arbiter, space, data):
        tipo = self._identificar_parte(arbiter)
        if tipo == COLL_PIE_IZQ: self.c_izq_counter += 1
        elif tipo == COLL_PIE_DER: self.c_der_counter += 1
        return True

    def _cb_separate(self, arbiter, space, data):
        tipo = self._identificar_parte(arbiter)
        if tipo == COLL_PIE_IZQ: self.c_izq_counter -= 1
        elif tipo == COLL_PIE_DER: self.c_der_counter -= 1
        return True

    def _cb_post_solve(self, arbiter, space, data):
        tipo = self._identificar_parte(arbiter)
        imp = arbiter.total_impulse.length
        if tipo == COLL_PIE_IZQ: self.impacto_izq = max(self.impacto_izq, imp)
        elif tipo == COLL_PIE_DER: self.impacto_der = max(self.impacto_der, imp)
        elif tipo == COLL_TORSO: self.impacto_torso = max(self.impacto_torso, imp)
        elif tipo == COLL_MUSLO: self.impacto_muslo = max(self.impacto_muslo, imp)
        return True

    # --- V9.1: LÓGICA DEL MANAGER ---
    def _actualizar_manager(self):
        """El cerebro superior decide la velocidad basándose en el Lidar."""
        # Si el obstáculo está cerca (menos del 40% del rayo de 700px = 280px)
        if self.dist_lidar_lejos < 0.4:
            # Orden: FRENAR / PREPARAR SALTO
            self.goal_velocity = 3.0 
        else:
            # Orden: SPRINT
            self.goal_velocity = 8.5 

    def step(self, action):
        self.tiempo += 1
        self.impacto_izq = 0.0; self.impacto_der = 0.0
        self.impacto_torso = 0.0; self.impacto_muslo = 0.0
        
        # PUSH
        if self.tiempo % 150 == 0 and np.random.rand() < 0.6: 
            fuerza_x = np.random.uniform(-5000, 7000) 
            self.torso_body.apply_impulse_at_local_point((fuerza_x, 0), (0,0))
            self.push_visual = 10 
        if self.push_visual > 0: self.push_visual -= 1

        # Parkour
        if self.torso_body.position.x > self.next_obstacle_x - 700:
            self._generar_obstaculo(self.next_obstacle_x)
            self.next_obstacle_x += np.random.randint(500, 1000) 

        for obs in self.lista_obstaculos[:]:
            if obs.position.x < self.torso_body.position.x - 1500:
                self.space.remove(obs, *obs.shapes) 
                self.lista_obstaculos.remove(obs)        

        # Motores
        action = np.clip(action, -1.0, 1.0)
        alpha = 0.6
        smoothed_action = (1 - alpha) * self.prev_action + alpha * action
        
        v_max = 9.0 
        motores = [self.pierna_izq['m_cadera'], self.pierna_izq['m_rodilla'],
                   self.pierna_der['m_cadera'], self.pierna_der['m_rodilla']]
        for i, m in enumerate(motores): m.rate = float(smoothed_action[i]) * v_max

        # FÍSICA
        for _ in range(4): self.space.step(1.0 / 240.0)
        
        # --- V9.1: ACTUALIZACIÓN DEL MANAGER (CADA 30 PASOS = 0.5s) ---
        # Más frecuente que 100 para mejor reactividad
        if self.tiempo % 30 == 0:
            self._actualizar_manager()

        # REWARD
        vel_x = self.torso_body.velocity.x
        angle = self.torso_body.angle
        pos_y = self.torso_body.position.y
        
        reward = 0.0
        
        # 1. RECOMPENSA HRL (OBEDIENCIA)
        # En lugar de premiar velocidad pura, premiamos acercarse a la META
        # Error de velocidad
        vel_error = abs(vel_x - self.goal_velocity)
        
        # Si cumple la orden (margen de error 1.0), gana puntos. Si no, penalización suave.
        if vel_error < 1.0:
            reward += 0.5 # Buen chico, vas a la velocidad correcta
        else:
            reward -= vel_error * 0.05 # Penalización por desobediencia
            
        # Incentivo base para que no se detenga si la meta es alta
        reward += vel_x * 0.01 

        if pos_y < 490: reward += 0.5 
        else: reward -= 1.0 
        
        reward -= abs(angle) * 0.8
        reward -= np.sum(np.square(smoothed_action)) * 0.005 

        terminated = False
        if self.impacto_muslo > 10.0: reward -= 20.0; terminated = True 
        if (self.impacto_izq + self.impacto_der) > 6000: reward -= 1.0 
        if self.impacto_torso > 200: reward -= 5.0
            
        is_contact_izq = self.c_izq_counter > 0
        is_contact_der = self.c_der_counter > 0
        if is_contact_izq and not self.prev_contact_izq: reward += 1.0 
        if is_contact_der and not self.prev_contact_der: reward += 1.0
        self.prev_contact_izq = is_contact_izq; self.prev_contact_der = is_contact_der
        self.prev_action = smoothed_action

        truncated = False
        if pos_y > 510 or abs(angle) > 1.2: reward -= 10.0; terminated = True
        if self.tiempo > 2000: truncated = True

        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        rb = self.torso_body
        d_suelo, n_suelo_x, n_suelo_y = self._raycast(pymunk.Vec2d(0,0), pymunk.Vec2d(0, 150), stabilized=False)
        d_lejos, _, _ = self._raycast(pymunk.Vec2d(0,0), pymunk.Vec2d(700, 80), stabilized=True)
        d_salto, _, _ = self._raycast(pymunk.Vec2d(0,0), pymunk.Vec2d(150, 60), stabilized=False)
        
        # Guardamos la lectura para el Manager
        self.dist_lidar_lejos = d_lejos 

        ang_muslo_izq = self.pierna_izq['muslo_body'].angle - rb.angle
        ang_rodilla_izq = self.pierna_izq['pie'].angle - self.pierna_izq['muslo_body'].angle
        ang_muslo_der = self.pierna_der['muslo_body'].angle - rb.angle
        ang_rodilla_der = self.pierna_der['pie'].angle - self.pierna_der['muslo_body'].angle

        imp_muslo_norm = min(self.impacto_muslo / 5000.0, 1.0)
        imp_torso_norm = min(self.impacto_torso / 5000.0, 1.0)

        # Normalizamos la meta para la red neuronal (0.0 a 1.0 aprox)
        meta_norm = self.goal_velocity / 9.0

        obs = [
            (rb.position.y - 450.0) / 100.0, np.sin(rb.angle), np.cos(rb.angle), 
            rb.velocity.x / 100.0, rb.velocity.y / 100.0,
            self.pierna_izq['m_cadera'].rate / 9.0, self.pierna_izq['m_rodilla'].rate / 9.0,
            self.pierna_der['m_cadera'].rate / 9.0, self.pierna_der['m_rodilla'].rate / 9.0,
            1.0 if self.c_izq_counter > 0 else 0.0, 1.0 if self.c_der_counter > 0 else 0.0,
            d_suelo, n_suelo_x, n_suelo_y, d_lejos, d_salto, 
            self.impacto_izq/5000.0, self.impacto_der/5000.0,
            ang_muslo_izq, ang_rodilla_izq, ang_muslo_der, ang_rodilla_der,
            self.prev_action[0], self.prev_action[1], self.prev_action[2], self.prev_action[3],
            self.grav_y / 1000.0, 
            imp_muslo_norm,       
            imp_torso_norm,
            
            # --- V9.1: INPUT 30 (LA ORDEN) ---
            meta_norm 
        ]
        return np.array(obs, dtype=np.float32)

    def _raycast(self, start, end, stabilized=False):
        p_start = self.torso_body.position + start.rotated(self.torso_body.angle)
        if stabilized: p_end = self.torso_body.position + end
        else: p_end = self.torso_body.position + end.rotated(self.torso_body.angle)
        f = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 2) 
        res = self.space.segment_query_first(p_start, p_end, 1, f)
        if res: return res.alpha, res.normal.x, res.normal.y 
        return 1.0, 0.0, -1.0 

    def _crear_escenario(self):
        suelo = pymunk.Segment(self.space.static_body, (-2000, 550), (500000, 550), 5.0)
        suelo.friction = 1.0; suelo.collision_type = COLL_SUELO 
        self.space.add(suelo)

    def _crear_robot(self, x, y, masa):
        self.torso_body = pymunk.Body(masa, pymunk.moment_for_box(masa, (40, 60)))
        self.torso_body.position = (x, y)
        shape = pymunk.Poly.create_box(self.torso_body, (40, 60))
        shape.filter = pymunk.ShapeFilter(group=2, categories=2) 
        shape.friction = 0.5
        shape.collision_type = COLL_TORSO 
        self.space.add(self.torso_body, shape)
        self.pierna_izq = self._crear_pierna(x, y+30, COLL_PIE_IZQ, self.friccion_dr) 
        self.pierna_der = self._crear_pierna(x, y+30, COLL_PIE_DER, self.friccion_dr) 

    def _crear_pierna(self, x, y, c_type_pie, friccion):
        m_muslo = 2.0; muslo = pymunk.Body(m_muslo, pymunk.moment_for_box(m_muslo, (12, 45)))
        muslo.position = (x, y + 25)
        m_shape = pymunk.Poly.create_box(muslo, (12, 45))
        m_shape.filter = pymunk.ShapeFilter(group=2, categories=2)
        m_shape.collision_type = COLL_MUSLO 
        m_shape.friction = friccion 
        pivot = pymunk.PivotJoint(self.torso_body, muslo, (0, 25), (0, -20))
        motor = pymunk.SimpleMotor(self.torso_body, muslo, 0)
        motor.max_force = 120000 
        limite = pymunk.RotaryLimitJoint(self.torso_body, muslo, -1.2, 0.4)
        self.space.add(muslo, m_shape, pivot, motor, limite)
        panto = pymunk.Body(1.0, pymunk.moment_for_box(1.0, (10, 45)))
        panto.position = (x, y + 65)
        p_shape = pymunk.Poly.create_box(panto, (10, 45))
        p_shape.filter = pymunk.ShapeFilter(group=2, categories=2)
        p_shape.friction = friccion 
        p_shape.collision_type = c_type_pie 
        pivot_r = pymunk.PivotJoint(muslo, panto, (0, 20), (0, -20))
        motor_r = pymunk.SimpleMotor(muslo, panto, 0)
        motor_r.max_force = 120000 
        limite_r = pymunk.RotaryLimitJoint(muslo, panto, 0.0, 2.2) 
        self.space.add(panto, p_shape, pivot_r, motor_r, limite_r)
        return {'m_cadera': motor, 'm_rodilla': motor_r, 'muslo_body': muslo, 'pie': panto}

    def _generar_obstaculo(self, x):
        h = np.random.randint(20, 45); w = np.random.randint(30, 60)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (x, 550 - (h/2))
        shape = pymunk.Poly.create_box(body, (w, h))
        shape.friction = 1.5 
        shape.collision_type = COLL_OBSTACULO; shape.color = (200, 50, 50, 255)
        self.space.add(body, shape); self.lista_obstaculos.append(body)

    def _render_frame(self):
        offset_x = -self.torso_body.position.x + 300
        self.draw_options.transform = pymunk.Transform.translation(offset_x, 0)
        self.screen.fill((240, 240, 240))
        self.space.debug_draw(self.draw_options)
        
        st = self.torso_body.position; end = st + pymunk.Vec2d(700, 80) 
        st_scr = st + pymunk.Vec2d(offset_x, 0); end_scr = end + pymunk.Vec2d(offset_x, 0)
        try: pygame.draw.line(self.screen, (0, 255, 0), (st_scr.x, st_scr.y), (end_scr.x, end_scr.y), 2)
        except: pass

        txt_vel = self.font.render(f"Vel: {self.torso_body.velocity.x:.1f} | META: {self.goal_velocity:.1f}", True, (50, 50, 50))
        self.screen.blit(txt_vel, (20, 20))
        txt_grav = self.font.render(f"Grav: {self.grav_y:.0f} | Fric: {self.friccion_dr:.1f}", True, (50, 50, 50))
        self.screen.blit(txt_grav, (20, 45))
        pygame.display.flip(); self.clock.tick(60)

# ==========================================
# TRAINING (HRL + SAC)
# ==========================================
def make_env(): return SonsoEnv(render_mode=None)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs("logs", exist_ok=True); os.makedirs("models", exist_ok=True)
    
    print("="*60)
    print(f"🧠 SONSO IA: V9.1 {NOMBRE_MODELO} (SMART MANAGER)")
    print("   -> HRL: Manager sets Speed Goals based on Lidar")
    print("   -> Algo: SAC (High Batch Size for Hierarchy)")
    print("   -> Input: 30 Neurons (Goal Aware)")
    print("="*60)

    n_procs = 4 
    env = SubprocVecEnv([make_env for _ in range(n_procs)])
    env = VecMonitor(env, filename=f"logs/{NOMBRE_MODELO}")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4,       
        buffer_size=600_000,      # Aumentado para HRL
        batch_size=512,           # Estabilidad jerárquica
        ent_coef='auto',          
        tau=0.005,                
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1, 
        device="auto",
        tensorboard_log="./tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f'./models/{NOMBRE_MODELO}/', name_prefix=NOMBRE_MODELO)

    print(f">> INICIANDO ENTRENAMIENTO JERÁRQUICO...")
    try:
        model.learn(total_timesteps=3_000_000, callback=checkpoint_callback, log_interval=4)
        model.save(NOMBRE_MODELO); env.save(f"{NOMBRE_MODELO}_vecnorm.pkl")
        print(">> ENTRENAMIENTO FINALIZADO.")
    except KeyboardInterrupt:
        model.save(NOMBRE_MODELO); env.save(f"{NOMBRE_MODELO}_vecnorm.pkl")
    finally:
        env.close()