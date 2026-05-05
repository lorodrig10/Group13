import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SHOW_PRINTS = True
PRINT_FREQ = 1000
SENSOR_FREQ = 20 
GO_STRAIGHT_THRESHOLD = 2 / 100
SKY_REGION_RATIO = 0.4 #allow to modify the % of height seen from the sky (the smaller the more high we see)
BINOCULAR_OVERLAP_RATIO = 0.1 #to avoid looking at the same region with both eyes, which can cause confusion in the obstacle detection
EXTERNAL_VISON_RATIO = 0.3  # to avoid looking at fare left and far right, which are less relevant for obstacle detection
SWEEP_HEIGHT = 20               # Hauteur (en pixels) de chaque bande analysée
MIN_GRASS_WIDTH = 35            # Largeur minimum (en pixels) pour considérer qu'il y a un obstacle

class State(Enum):
    FOLLOW_SCENT = auto()
    AVOID_OBSTACLE = auto()
    DODGE_DRAGON = auto()


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self.state = State.FOLLOW_SCENT
        self.drive = np.array([0.0, 0.0])
        self.show_prints = False

        self.odor_smooth = None
        self.alpha = 0.0005

        # self.obstacle_threshold = 0.015
        # self.min_green_height_ratio = 0.10

        #UNCOMMENT TO ACTIVATE TIME FOR DETECTION
        #self.avoid_duration = 300
        #self.avoid_timer = 0

    def step(self, sim: MiniprojectSimulation, step):
        self.show_prints = SHOW_PRINTS and step % PRINT_FREQ == 0

        if step % SENSOR_FREQ == 0:
            raw_olfaction = sim.get_olfaction(sim.fly.name)

            if self.odor_smooth is None:
                self.odor_smooth = raw_olfaction.copy()
            else:
                self.odor_smooth = (
                    (1 - self.alpha) * self.odor_smooth
                    + self.alpha * raw_olfaction
                )

            left_h, right_h, obstacle_found = self.detect_obstacle(sim, visualize=self.show_prints)

            if self.show_prints:
                print(f"Vision obstacle scores: L={left_h:.4f}, R={right_h:.4f}")

            if obstacle_found:
                self.state = State.AVOID_OBSTACLE
            else:
                self.state = State.FOLLOW_SCENT

            if self.state == State.FOLLOW_SCENT:
                self.drive = self.follow_scent()
            elif self.state == State.AVOID_OBSTACLE:
                self.drive = self.avoid_obstacle(left_h, right_h)
            else:
                self.drive = np.array([0.0, 0.0])

        if step > 0 and self.show_prints:
            fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
            plt.figure(figsize=(10, 4))
            plt.imshow(fly_vision)
            plt.title(f"Vision step {step} — {self.state.name}")
            plt.axis("off")
            plt.show()

        joint_angles, adhesion = self.turning_controller.step(self.drive)

        return joint_angles, adhesion

    # def detect_obstacle(self, sim):
    #     """
    #     Utilise directement les images RGB de sim.get_raw_vision().
    #     """

    #     eye_imgs = sim.get_raw_vision(sim.fly.name)
    #     scores = []

    #     for img in eye_imgs:
    #         img = np.asarray(img)

    #         if img.max() <= 1.0:
    #             img = img * 255.0

    #         h, w, c = img.shape

    #         sky_region = img[: int(h * self.sky_region_ratio), :, :]

    #         r = sky_region[:, :, 0].astype(float)
    #         g = sky_region[:, :, 1].astype(float)
    #         b = sky_region[:, :, 2].astype(float)

    #         green_mask = (
    #             (g > 90)
    #             & (g > r * 1.25)
    #             & (g > b * 1.05)
    #         )

    #         column_green_ratio = green_mask.mean(axis=0)
    #         tall_green_columns = column_green_ratio > self.min_green_height_ratio

    #         score = tall_green_columns.mean()
    #         scores.append(score)

    #     return scores[0], scores[1]
    # def detect_obstacle(self, sim):
    #     """
    #     Utilise directement les images RGB de sim.get_raw_vision() avec 
    #     des restrictions de champ de vision asymétriques.
    #     """
    #     eye_imgs = sim.get_raw_vision(sim.fly.name)
    #     scores = []

    #     for i, img in enumerate(eye_imgs):
    #         img = np.asarray(img)

    #         if img.max() <= 1.0:
    #             img = img * 255.0

    #         h, w, c = img.shape

    #         if i == 0:
    #             start_w = int(w * EXTERNAL_VISON_RATIO)
    #             end_w = int(w * (1 - BINOCULAR_OVERLAP_RATIO))
    #         else:
    #             start_w = int(w * BINOCULAR_OVERLAP_RATIO)
    #             end_w = int(w * (1 - EXTERNAL_VISON_RATIO))

    #         target_region = img[: int(h * SKY_REGION_RATIO), start_w:end_w, :]

    #         r = target_region[:, :, 0].astype(float)
    #         g = target_region[:, :, 1].astype(float)
    #         b = target_region[:, :, 2].astype(float)

    #         green_mask = (
    #             (g > 90)
    #             & (g > r * 1.25)
    #             & (g > b * 1.05)
    #         )

    #         column_green_ratio = green_mask.mean(axis=0)
    #         tall_green_columns = column_green_ratio > self.min_green_height_ratio

    #         if len(tall_green_columns) > 0:
    #             score = tall_green_columns.mean()
    #         else:
    #             score = 0.0
                
    #         scores.append(score)

    #     return scores[0], scores[1]
    def detect_obstacle(self, sim, visualize=False):
        """
        Détecte les obstacles verts en balayant l'image de haut en bas.
        Retourne la "hauteur" (coordonnée Y depuis le haut) de la première 
        bande contenant suffisamment de pixels verts.
        """
        eye_imgs = sim.get_raw_vision(sim.fly.name)
        
        # Pour stocker les résultats finaux
        heights = []
        widths = []
        
        # Dictionnaire pour stocker les données nécessaires à la visualisation
        debug_info = {'imgs': [], 'masks': [], 'boxes': []}

        for i, img in enumerate(eye_imgs):
            img = np.asarray(img)

            if img.max() <= 1.0:
                img = img * 255.0

            h, w, c = img.shape
            target_h = int(h * SKY_REGION_RATIO)

            # Détermination des limites (Cadre d'observation)
            if i == 0:  #left eye
                start_w = int(w * EXTERNAL_VISON_RATIO)
                end_w = int(w * (1 - BINOCULAR_OVERLAP_RATIO))
            else:       #right eye
                start_w = int(w * BINOCULAR_OVERLAP_RATIO)
                end_w = int(w * (1 - EXTERNAL_VISON_RATIO))

            target_region = img[:target_h, start_w:end_w, :]

            r = target_region[:, :, 0].astype(float)
            g = target_region[:, :, 1].astype(float)
            b = target_region[:, :, 2].astype(float)

            green_mask = (
                (g > 90)
                & (g > r * 1.25)
                & (g > b * 1.05)
            )

            obstacle_y = -1     # -1 = no obstacle found
            obstacle_width = 0

            for y in range(0, target_h, SWEEP_HEIGHT): #sweep the mask from top to bottom with a step of SWEEP_HEIGHT
                band = green_mask[y:min(y + SWEEP_HEIGHT, target_h), :]
                
                band_width = np.sum(np.any(band, axis=0)) #counts how many green pixel has the current band

                # Si la largeur dépasse notre seuil, on a trouvé le haut de l'obstacle !
                if band_width >= MIN_GRASS_WIDTH:
                    obstacle_y = y
                    obstacle_width = band_width
                    break # On arrête le balayage pour garder la position la plus haute

            heights.append(obstacle_y)
            widths.append(obstacle_width)
            
            if visualize:
                debug_info['imgs'].append(img.astype(np.uint8))
                debug_info['masks'].append(green_mask)
                debug_info['boxes'].append((0, start_w, target_h, end_w - start_w)) # y, x, h, w

        # Appel à la fonction de visualisation si demandé
        obstacle_found = any(h != -1 for h in heights)
        if visualize and obstacle_found:
            self.visualize_detection(debug_info, heights)

        return heights[0], heights[1], obstacle_found

    def visualize_detection(self, debug_info, heights):
        """
        Affiche 4 graphiques : Les 2 images originales avec le cadre rouge, 
        et les 2 masques verts avec une ligne/bande bleue indiquant la détection.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        titles = ["Œil Gauche", "Œil Droit"]
        
        for i in range(2):
            img = debug_info['imgs'][i]
            mask = debug_info['masks'][i]
            box_y, box_x, box_h, box_w = debug_info['boxes'][i]
            detected_y = heights[i]

            # --- Ligne 1 : Image originale + Cadre ---
            ax_img = axes[0, i]
            ax_img.imshow(img)
            ax_img.set_title(f"{titles[i]} - Image Originale")
            
            # Dessin du rectangle rouge pour montrer la zone observée
            rect = patches.Rectangle((box_x, box_y), box_w, box_h, 
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax_img.add_patch(rect)
            
            # --- Ligne 2 : Masque Vert + Ligne de détection ---
            ax_mask = axes[1, i]
            ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(f"{titles[i]} - Masque Vert (Rogné)")
            
            # Si un obstacle a été trouvé, on dessine la détection
            if detected_y != -1:
                # Dessin de la bande bleue translucide sur le masque
                rect_band = patches.Rectangle((0, detected_y), box_w, SWEEP_HEIGHT, 
                                              linewidth=0, facecolor='blue', alpha=0.4)
                ax_mask.add_patch(rect_band)
                
                # Ligne d'indication exacte du haut
                ax_mask.axhline(detected_y, color='blue', linestyle='-', linewidth=2, 
                                label=f"Haut détecté (Y={detected_y})")
                
                # (Optionnel) On reporte aussi la ligne bleue sur l'image originale
                ax_img.hlines(y=detected_y, xmin=box_x, xmax=box_x+box_w, colors='blue', linewidth=2)
                
                ax_mask.legend(loc="lower right")
            else:
                ax_mask.text(box_w/2, box_h/2, "Rien détecté", color='red', 
                             ha='center', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def avoid_obstacle(self, left_h, right_h):
        if self.show_prints:
            print(f"AVOID_OBSTACLE: L={left_h:.4f}, R={right_h:.4f}")

        if left_h < right_h:
            if self.show_prints:
                print("Obstacle à gauche → tourner à droite")
            return np.array([1.5, 0.2])
        else:
            if self.show_prints:
                print("Obstacle à droite → tourner à gauche")
            return np.array([0.2, 1.5])

    def follow_scent(self):
        left_odor_a = self.odor_smooth[0, 0]
        left_odor_b = self.odor_smooth[2, 0]
        right_odor_a = self.odor_smooth[1, 0]
        right_odor_b = self.odor_smooth[3, 0]

        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b

        eps = 1e-8
        ratio = abs(left_signal) / (abs(right_signal) + eps)

        if self.show_prints:
            print(f"Olfaction: {self.odor_smooth}")
            print(f"Left signal: {left_signal}, Right signal: {right_signal}")
            print(f"ratio: {ratio}")

        if abs(ratio - 1) < GO_STRAIGHT_THRESHOLD:
            if self.show_prints:
                print("FOLLOW_SCENT: going straight")
            return np.array([2.0, 2.0])

        elif left_signal > right_signal:
            if self.show_prints:
                print("FOLLOW_SCENT: turning left")
            return np.array([0.2, 1.0])

        else:
            if self.show_prints:
                print("FOLLOW_SCENT: turning right")
            return np.array([1.0, 0.2])