import pygame
import sys
import math
import os
import cv2
import numpy as np
import mediapipe as mp
from pygame.locals import *

WIDTH, HEIGHT = 1280, 720
FONT_SIZE = 18
COLORS = {
    'background': (30, 30, 30),
    'text': (255, 255, 255),
    'backbone': (200, 25, 25),    
    'other_atoms': (100, 100, 255),
    'bond': (50, 50, 50),
    'fog': (40, 40, 40),
    'depth_gradient': [(i, i, i) for i in range(0, 256, 8)],
    'ui_bg': (50, 50, 50, 50)
}

class PDBViewer:
    def __init__(self, pdb_file):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3D PDB Viewer 🔮")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', FONT_SIZE, bold=True)        
        try:
            self.atoms, self.bonds = self.parse_pdb(pdb_file)
        except Exception as e:
            print(f"Error loading PDB file: {str(e)}")
            pygame.quit()
            sys.exit(1)
        
        self.view = {
            'angle_x': 0,          
            'angle_y': 0,          
            'distance': 300,       
            'fov': 1000,
            'dragging': False,
            'near_clip': 1,
            'far_clip': 3000
        }
        
        self.settings = {
            'show_depth': False,
            'fog_strength': 0.7,
            'atom_size_range': (2, 6),
            'bond_width_range': (1, 3),
            'use_white_bg': False,
            'show_ui': True
        }
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            pygame.quit()
            sys.exit(1)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.filtered_angle_x = 0
        self.filtered_angle_y = 0
        self.filtered_distance = self.view['distance']
        
        self.alpha_rot = 0.3   
        self.alpha_dist = 0.2  

    def parse_pdb(self, pdb_file):
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        atoms = []
        bonds = []
        
        residue_map = {}
        prev_res = None
        
        with open(pdb_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith("ATOM"):
                    try:
                        atom_name = line[12:16].strip()
                        chain = line[21]
                        res_num = line[22:26].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Invalid PDB format at line {line_num}") from e
                    
                    atom_data = {
                        'x': x, 
                        'y': y, 
                        'z': z,
                        'atom_name': atom_name,
                        'chain': chain,
                        'res_num': res_num
                    }
                    idx = len(atoms)
                    atoms.append(atom_data)
                    
                    res_key = (chain, res_num)
                    if res_key not in residue_map:
                        residue_map[res_key] = {'N': None, 'CA': None, 'C': None}
                    
                    if atom_name in ['N', 'CA', 'C']:
                        residue_map[res_key][atom_name] = idx
                        
                        if atom_name == 'N':
                            if residue_map[res_key]['CA'] is not None:
                                bonds.append((idx, residue_map[res_key]['CA']))
                        elif atom_name == 'CA':
                            if residue_map[res_key]['N'] is not None:
                                bonds.append((residue_map[res_key]['N'], idx))
                            if residue_map[res_key]['C'] is not None:
                                bonds.append((idx, residue_map[res_key]['C']))
                        elif atom_name == 'C':
                            if residue_map[res_key]['CA'] is not None:
                                bonds.append((residue_map[res_key]['CA'], idx))
                    

                    if prev_res and atom_name == 'N':
                        if residue_map[prev_res].get('C') is not None:
                            bonds.append((residue_map[prev_res]['C'], idx))
                    
                    if atom_name == 'C':
                        prev_res = res_key
        
        return atoms, bonds

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                self.view['dragging'] = True
                pygame.mouse.get_rel()
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                self.view['dragging'] = False
            elif event.type == MOUSEMOTION and self.view['dragging']:
                dx, dy = pygame.mouse.get_rel()
                self.view['angle_y'] += dx * 0.3
                self.view['angle_x'] += dy * 0.3
            elif event.type == MOUSEWHEEL:
                self.view['distance'] = max(10, self.view['distance'] - event.y * 20)
            elif event.type == KEYDOWN:
                if event.key == K_d:
                    self.settings['show_depth'] = not self.settings['show_depth']
                elif event.key == K_f:
                    self.settings['fog_strength'] = 0.0 if self.settings['fog_strength'] > 0 else 0.7
                elif event.key == K_u:
                    self.settings['show_ui'] = not self.settings['show_ui']
                elif event.key == K_c:
                    self.settings['use_white_bg'] = not self.settings['use_white_bg']
                elif event.key == K_ESCAPE:
                    return False
        return True

    def capture_and_process_camera(self):

        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            h, w = frame.shape[:2]
            
            
            wrist       = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]            # 0
            index_mcp   = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP] # 5
            pinky_mcp   = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]        # 17
            index_tip   = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP] # 8
            thumb_tip   = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]        # 4
            
            px = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
            py = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
            
            norm_x = (px - 0.5) 
            norm_y = (py - 0.5)
            
            desired_angle_y = norm_x * 360.0
            desired_angle_x = -norm_y * 360.0  

            self.filtered_angle_x = (1 - self.alpha_rot)*self.filtered_angle_x + self.alpha_rot*desired_angle_x
            self.filtered_angle_y = (1 - self.alpha_rot)*self.filtered_angle_y + self.alpha_rot*desired_angle_y
            
            self.view['angle_x'] = self.filtered_angle_x
            self.view['angle_y'] = self.filtered_angle_y
            
            ix, iy = index_tip.x, index_tip.y
            tx, ty = thumb_tip.x, thumb_tip.y
            pinch_dist = math.dist((ix, iy), (tx, ty))
            
            print(f"Pinch dist: {pinch_dist:.2f}", end="  ")
            
            dist_min, dist_max = 0.05, 0.40
            zoom_near, zoom_far = 100, 600
            
            pinch_clamped = max(dist_min, min(dist_max, pinch_dist))
            ratio = (pinch_clamped - dist_min) / (dist_max - dist_min)
            desired_dist = zoom_near + (zoom_far - zoom_near) * ratio
            
            print(f"=> Zoom dist: {desired_dist:.2f}")
            
            self.filtered_distance = (1 - self.alpha_dist)*self.filtered_distance + self.alpha_dist*desired_dist
            self.view['distance'] = self.filtered_distance
        
        show_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        show_frame = cv2.resize(show_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        return pygame.surfarray.make_surface(show_frame.swapaxes(0,1))

    def project_point(self, x, y, z):
        
        x_rot = x * math.cos(math.radians(self.view['angle_y'])) + z * math.sin(math.radians(self.view['angle_y']))
        z_rot = z * math.cos(math.radians(self.view['angle_y'])) - x * math.sin(math.radians(self.view['angle_y']))
        
        y_rot = y * math.cos(math.radians(self.view['angle_x'])) - z_rot * math.sin(math.radians(self.view['angle_x']))
        z_final = y * math.sin(math.radians(self.view['angle_x'])) + z_rot * math.cos(math.radians(self.view['angle_x']))
        
        z_clip = z_final + self.view['distance']
        if z_clip < 1:
            return None, None, None
        
        factor = self.view['fov'] / z_clip
        px = int(x_rot * factor + WIDTH / 2)
        py = int(-y_rot * factor + HEIGHT / 2)
        return px, py, z_clip

    def draw(self, camera_surf):
    
        if self.settings['use_white_bg']:
            self.screen.fill((255, 255, 255))
        else:
            if camera_surf is not None:
                self.screen.blit(camera_surf, (0, 0))
            else:
                self.screen.fill(COLORS['background'])   

        projected = []
        min_depth = float('inf')
        max_depth = -float('inf')
        
        for i, atom in enumerate(self.atoms):
            px, py, depth = self.project_point(atom['x'], atom['y'], atom['z'])
            if px is None:
                continue
            projected.append((px, py, depth, i))
            min_depth = min(min_depth, depth)
            max_depth = max(max_depth, depth)
        
        if not projected:
            return
        
        proj_dict = {p[3]: p for p in projected}
        
        for (a1, a2) in self.bonds:
            if a1 in proj_dict and a2 in proj_dict:
                x1, y1, d1, idx1 = proj_dict[a1]
                x2, y2, d2, idx2 = proj_dict[a2]
                avg_depth = (d1 + d2) / 2
                t = (avg_depth - min_depth) / (max_depth - min_depth + 1e-6)
                
                width = self.lerp(*self.settings['bond_width_range'], t)
                color = COLORS['bond']
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), int(width))
        
        projected_sorted = sorted(projected, key=lambda x: -x[2])
        
        for px, py, depth, idx in projected_sorted:
            atom = self.atoms[idx]
            t = (depth - min_depth) / (max_depth - min_depth + 1e-6)
            
            if atom['atom_name'] in ['N','CA','C']:
                base_color = COLORS['backbone']
            else:
                base_color = COLORS['other_atoms']
            
            color = self.apply_fog(base_color, t)
            size = self.lerp(*self.settings['atom_size_range'], 1 - t)
            
            pygame.draw.circle(self.screen, color, (px, py), int(size))
        
        if self.settings['show_ui']:
            self.draw_ui(min_depth, max_depth)
        
        pygame.display.flip()

    def apply_fog(self, color, t):
        fog_amount = t * self.settings['fog_strength']
        return tuple(
            int(c*(1 - fog_amount) + COLORS['fog'][i]*fog_amount)
            for i, c in enumerate(color)
        )

    def lerp(self, a, b, t):
        return a + (b-a)*t

    def draw_ui(self, min_depth, max_depth):
        if self.settings['show_depth']:
            legend = pygame.Surface((30, 200), pygame.SRCALPHA)
            for y in range(200):
                tt = y/200
                cc = COLORS['depth_gradient'][int(tt*(len(COLORS['depth_gradient'])-1))]
                pygame.draw.line(legend, cc, (0, y), (30, y))
            self.screen.blit(legend, (WIDTH-40, 20))
            self.draw_text(WIDTH-80, 20, f"{max_depth:.0f}", COLORS['text'])
            self.draw_text(WIDTH-80, 210, f"{min_depth:.0f}", COLORS['text'])

        lines = [
            "Keyboard/Mouse:",
            " - LMB Drag => Rotate",
            " - Mouse Wheel => Zoom",
            " - D => Depth view toggle",
            " - F => Fog toggle",
            " - U => UI toggle",
            " - C => White Background",
            " - ESC => Quit"
        ]
        
        panel_height = 20*len(lines) + 10
        panel = pygame.Surface((420, panel_height), pygame.SRCALPHA)
        panel.fill(COLORS['ui_bg'])
        for i, text_line in enumerate(lines):
            self.draw_text(10, 10+i*20, text_line, COLORS['text'], panel)
        self.screen.blit(panel, (20, 20))

    def draw_text(self, x, y, text, color, surface=None):
        surf = surface or self.screen
        text_surf = self.font.render(text, True, color)
        surf.blit(text_surf, (x, y))

    def run(self):
        """Main loop."""
        try:
            running = True
            while running:
                running = self.handle_events()
                camera_surf = self.capture_and_process_camera()
                self.draw(camera_surf)
                self.clock.tick(30)
        except Exception as e:
            print("Runtime error:", e)
        finally:
            self.cap.release()
            pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdb_path = script_dir + "/RNR.pdb" #Place your path here
    
    if not os.path.exists(pdb_path):
        print(f"PDB file not found: {pdb_path}")
        sys.exit(1)
    
    viewer = PDBViewer(pdb_path)
    viewer.run()
