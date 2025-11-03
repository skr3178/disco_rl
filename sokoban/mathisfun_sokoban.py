# mathisfun_sokoban.py
import json, pathlib, gymnasium as gym, numpy as np
from gymnasium import spaces

WALL, FLOOR, BOX, TARGET, PLAYER, BOX_ON_TARGET, PLAYER_ON_TARGET = 0,1,2,3,4,5,6

class MathIsFunSokoban(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=300):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.window_closed = False  # Track if window was closed
        self._fixed_level_idx = None

        # Load the 60 levels we downloaded
        path = pathlib.Path(__file__).with_name("levels_clean.json")
        with open(path) as f:
            raw = json.load(f)
        self.levels = [self._parse_ascii(lvl) for lvl in raw]

        # Find max dimensions across all levels for observation space
        max_h = max_w = 0
        for level_data in self.levels:
            h, w = len(level_data["grid"]), len(level_data["grid"][0])
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        self.max_h = max_h
        self.max_w = max_w
        self.scale = 640 // max(max_h, max_w)  # Increased from 160 to 640 for better visibility
        obs_h, obs_w = max_h * self.scale, max_w * self.scale
        
        self.observation_space = spaces.Box(0, 255, shape=(obs_h, obs_w, 3), dtype=np.uint8)
        # 9 actions: 0=No-Op, 1-4=Push(Up,Down,Left,Right), 5-8=Move(Up,Down,Left,Right)
        self.action_space = spaces.Discrete(9)

    # ------------------------------------------------------------
    def _parse_ascii(self, s):
        lines = [ln for ln in s.split('\n') if ln]
        h, w = len(lines), max(len(ln) for ln in lines)
        grid = np.ones((h, w), dtype=int)   # default FLOOR
        player = boxes = targets = None
        for i, row in enumerate(lines):
            for j, c in enumerate(row):
                if c == '#': grid[i,j] = WALL
                elif c == '$': grid[i,j] = BOX; boxes = (boxes or []) + [[i,j]]
                elif c == '.': grid[i,j] = TARGET; targets = (targets or []) + [[i,j]]
                elif c == '@': grid[i,j] = PLAYER; player = [i,j]
                elif c == '*': grid[i,j] = BOX_ON_TARGET; boxes = (boxes or []) + [[i,j]]; targets = (targets or []) + [[i,j]]
                elif c == '+': grid[i,j] = PLAYER_ON_TARGET; player = [i,j]; targets = (targets or []) + [[i,j]]
        return {"grid":grid.tolist(), "player":player, "boxes":boxes, "targets":targets}
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        if self._fixed_level_idx is not None:
            options["level_idx"] = self._fixed_level_idx
        idx = options.get("level_idx", self.np_random.integers(len(self.levels)))
        self.level = self.levels[idx]
        self.grid = np.array(self.level["grid"], dtype=int)
        self.player = np.array(self.level["player"])
        self.step_count = 0
        return self._render_obs(), {}

    def step(self, action):
        # Action mapping
        # 0: No-Op
        # 1: Push Up, 2: Push Down, 3: Push Left, 4: Push Right
        # 5: Move Up, 6: Move Down, 7: Move Left, 8: Move Right
        push_dirs = {
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1),   # Right
        }
        move_dirs = {
            5: (-1, 0),  # Up
            6: (1, 0),   # Down
            7: (0, -1),  # Left
            8: (0, 1),   # Right
        }

        reward = 0.0
        did_move = False
        did_push = False
        pushed_on_target = False
        pushed_off_target = False

        # Handle No-Op
        if action == 0:
            self.step_count += 1
            solved = not np.any(self.grid == BOX)
            truncated = self.step_count >= self.max_steps
            # Per-step penalty
            reward -= 0.1
            if solved:
                reward += 10.0
            return self._render_obs(), reward, solved, truncated, {}

        # Determine direction and intended behavior
        if action in push_dirs:
            dx, dy = push_dirs[action]
            target_x, target_y = self.player[0] + dx, self.player[1] + dy
            # If adjacent is a box, try to push
            if self.grid[target_x, target_y] in (BOX, BOX_ON_TARGET):
                box_dest_x, box_dest_y = target_x + dx, target_y + dy
                # Check bounds
                if 0 <= box_dest_x < self.grid.shape[0] and 0 <= box_dest_y < self.grid.shape[1]:
                    # Destination must be free (not wall, not box)
                    if self.grid[box_dest_x, box_dest_y] not in (WALL, BOX, BOX_ON_TARGET):
                        # Compute on/off target reward signals
                        source_was_on_target = (self.grid[target_x, target_y] == BOX_ON_TARGET)
                        dest_is_target = (self.grid[box_dest_x, box_dest_y] == TARGET)

                        # Move the box
                        self.grid[box_dest_x, box_dest_y] = BOX_ON_TARGET if dest_is_target else BOX
                        self.grid[target_x, target_y] = TARGET if source_was_on_target else FLOOR

                        # Move the player into the vacated box cell
                        self.grid[self.player[0], self.player[1]] = (
                            TARGET if self.grid[self.player[0], self.player[1]] == PLAYER_ON_TARGET else FLOOR
                        )
                        self.player = np.array([target_x, target_y])
                        self.grid[target_x, target_y] = (
                            PLAYER_ON_TARGET if self.grid[target_x, target_y] == TARGET else PLAYER
                        )

                        did_move = True
                        did_push = True
                        if dest_is_target:
                            pushed_on_target = True
                        if source_was_on_target:
                            pushed_off_target = True
            # If no box to push, treat like a move in that direction
            else:
                # fall through as a move using same dx,dy
                nx, ny = target_x, target_y
                if self.grid[nx, ny] not in (WALL, BOX, BOX_ON_TARGET):
                    self.grid[self.player[0], self.player[1]] = (
                        TARGET if self.grid[self.player[0], self.player[1]] == PLAYER_ON_TARGET else FLOOR
                    )
                    self.player = np.array([nx, ny])
                    self.grid[nx, ny] = PLAYER_ON_TARGET if self.grid[nx, ny] == TARGET else PLAYER
                    did_move = True
        elif action in move_dirs:
            dx, dy = move_dirs[action]
            nx, ny = self.player[0] + dx, self.player[1] + dy
            if self.grid[nx, ny] not in (WALL, BOX, BOX_ON_TARGET):
                self.grid[self.player[0], self.player[1]] = (
                    TARGET if self.grid[self.player[0], self.player[1]] == PLAYER_ON_TARGET else FLOOR
                )
                self.player = np.array([nx, ny])
                self.grid[nx, ny] = PLAYER_ON_TARGET if self.grid[nx, ny] == TARGET else PLAYER
                did_move = True

        # Per-step bookkeeping
        self.step_count += 1
        solved = not np.any(self.grid == BOX)               # no plain BOX left
        truncated = self.step_count >= self.max_steps

        # Rewards per spec
        reward -= 0.1  # step penalty always
        if did_push:
            if pushed_on_target:
                reward += 1.0
            if pushed_off_target:
                reward -= 1.0
        if solved:
            reward += 10.0

        return self._render_obs(), reward, solved, truncated, {}

    # ------------------------------------------------------------
    def _render_obs(self):
        h, w = self.grid.shape
        # Use the same scale for all levels
        img_h, img_w = h*self.scale, w*self.scale
        # Start with gray background
        img = np.full((img_h, img_w, 3), 160, dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                y1, y2 = i*self.scale, (i+1)*self.scale
                x1, x2 = j*self.scale, (j+1)*self.scale
                tile = self.grid[i,j]
                
                if tile == WALL:
                    # Dark gray wall with darker border
                    img[y1:y2, x1:x2] = [50, 50, 50]
                    if self.scale > 4:
                        img[y1:y1+2, x1:x2] = [30, 30, 30]  # top
                        img[y2-2:y2, x1:x2] = [30, 30, 30]  # bottom
                        img[y1:y2, x1:x1+2] = [30, 30, 30]  # left
                        img[y1:y2, x2-2:x2] = [30, 30, 30]  # right
                        
                elif tile == FLOOR:
                    # Light floor
                    img[y1:y2, x1:x2] = [245, 245, 220]
                    
                elif tile == TARGET:
                    # Light floor with green target circle
                    img[y1:y2, x1:x2] = [245, 245, 220]
                    if self.scale > 8:
                        cy, cx = (y1+y2)//2, (x1+x2)//2
                        r = self.scale // 4
                        for dy in range(-r, r+1):
                            for dx in range(-r, r+1):
                                if dy*dy + dx*dx <= r*r:
                                    py, px = cy+dy, cx+dx
                                    if 0 <= py < img_h and 0 <= px < img_w:
                                        img[py, px] = [144, 238, 144]
                                        
                elif tile == BOX:
                    # Brown box with 3D effect
                    img[y1:y2, x1:x2] = [160, 82, 45]
                    if self.scale > 6:
                        # Lighter top-left (highlight)
                        sz = max(2, self.scale//8)
                        img[y1:y1+sz, x1:x2] = [200, 120, 80]
                        img[y1:y2, x1:x1+sz] = [200, 120, 80]
                        # Darker bottom-right (shadow)
                        img[y2-sz:y2, x1:x2] = [100, 50, 25]
                        img[y1:y2, x2-sz:x2] = [100, 50, 25]
                        # Border
                        img[y1:y1+1, x1:x2] = [80, 40, 20]
                        img[y2-1:y2, x1:x2] = [80, 40, 20]
                        img[y1:y2, x1:x1+1] = [80, 40, 20]
                        img[y1:y2, x2-1:x2] = [80, 40, 20]
                        
                elif tile == BOX_ON_TARGET:
                    # Box on target - greenish brown
                    img[y1:y2, x1:x2] = [180, 140, 70]
                    if self.scale > 6:
                        sz = max(2, self.scale//8)
                        img[y1:y1+sz, x1:x2] = [220, 180, 110]
                        img[y1:y2, x1:x1+sz] = [220, 180, 110]
                        img[y2-sz:y2, x1:x2] = [120, 90, 40]
                        img[y1:y2, x2-sz:x2] = [120, 90, 40]
                        img[y1:y1+1, x1:x2] = [90, 70, 30]
                        img[y2-1:y2, x1:x2] = [90, 70, 30]
                        img[y1:y2, x1:x1+1] = [90, 70, 30]
                        img[y1:y2, x2-1:x2] = [90, 70, 30]
                        
                elif tile == PLAYER:
                    # Light floor
                    img[y1:y2, x1:x2] = [245, 245, 220]
                    # Red player (circle-ish)
                    if self.scale > 8:
                        cy, cx = (y1+y2)//2, (x1+x2)//2
                        r = self.scale // 3
                        for dy in range(-r, r+1):
                            for dx in range(-r, r+1):
                                if dy*dy + dx*dx <= r*r:
                                    py, px = cy+dy, cx+dx
                                    if 0 <= py < img_h and 0 <= px < img_w:
                                        img[py, px] = [255, 50, 50]
                                        
                elif tile == PLAYER_ON_TARGET:
                    # Green target
                    img[y1:y2, x1:x2] = [245, 245, 220]
                    if self.scale > 8:
                        cy, cx = (y1+y2)//2, (x1+x2)//2
                        r = self.scale // 4
                        for dy in range(-r, r+1):
                            for dx in range(-r, r+1):
                                if dy*dy + dx*dx <= r*r:
                                    py, px = cy+dy, cx+dx
                                    if 0 <= py < img_h and 0 <= px < img_w:
                                        img[py, px] = [144, 238, 144]
                        # Red player on top
                        r = self.scale // 3
                        for dy in range(-r, r+1):
                            for dx in range(-r, r+1):
                                if dy*dy + dx*dx <= r*r:
                                    py, px = cy+dy, cx+dx
                                    if 0 <= py < img_h and 0 <= px < img_w:
                                        img[py, px] = [255, 50, 50]
        
        # Pad to observation space size
        obs_h, obs_w = self.observation_space.shape[:2]
        if img_h < obs_h or img_w < obs_w:
            padded = np.full((obs_h, obs_w, 3), 160, dtype=np.uint8)
            padded[:img_h, :img_w] = img
            return padded
        return img
    # ------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            import pygame
            if not hasattr(self, "screen") or not pygame.display.get_init():
                pygame.init()
                obs_h, obs_w = self.observation_space.shape[:2]
                self.screen = pygame.display.set_mode((obs_w, obs_h))
                pygame.display.set_caption("MathIsFun Sokoban")
                self.clock = pygame.time.Clock()
            
            # Handle pygame events (important for macOS!)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_closed = True
                    self.close()
                    return None
            
            if self.window_closed:
                return None
            
            surf = pygame.surfarray.make_surface(self._render_obs().swapaxes(0,1))
            self.screen.blit(surf, (0,0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return self._render_obs()

    def close(self):
        if hasattr(self, "screen"):
            import pygame
            pygame.quit()

    def set_level_idx(self, idx):
        self._fixed_level_idx = idx


# Auto-register the environment when this module is imported
from gymnasium.envs.registration import register

# Only register if not already registered
try:
    gym.envs.registration.env_specs.get("MathIsFunSokoban-v0")
except (AttributeError, KeyError):
    register(
        id="MathIsFunSokoban-v0",
        entry_point="mathisfun_sokoban:MathIsFunSokoban",
        max_episode_steps=300,
    )

