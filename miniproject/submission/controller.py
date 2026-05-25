import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SHOW_PRINTS = False
PRINT_FREQ = 10000
#for optimum run put back to 5 
SENSOR_FREQ = 10
GO_STRAIGHT_THRESHOLD = 2.5 / 100
SKY_REGION_RATIO = 0.42 #allow to modify the % of height seen from the sky (the smaller the more high we see)
BINOCULAR_OVERLAP_RATIO = 0.12 #to avoid looking at the same region with both eyes, -> might be ineffective
EXTERNAL_VISON_RATIO = 0.25  # to avoid looking at fare left and far right, which are less relevant for obstacle detection
SWEEP_HEIGHT = 5               # Hauteur (en pixels) de chaque bande analysée
MIN_GRASS_WIDTH = 30            # Largeur minimum (en pixels); slightly lower helps thin blades register earlier
AVOID_HOLD_STEPS = 48           # Continue evasive drive briefly after close grass leaves FOV
#of goes to shit put back at 22 instead of 100
OBSTACLE_ROW_CLOSE = 100         # Row from top of ROI — smaller means obstacle appears larger / closer
# Grass top row y_top <= limit => threat. Higher fraction => react earlier (larger limit).
OBSTACLE_THREAT_FRAC_OF_ROI = 0.76
# When olfaction says “go straight” (common near the banana), widen threat so grass lower in the sky ROI still triggers full avoidance.
OBSTACLE_THREAT_STRAIGHT_FRAC = 0.93
# Grass visible but below threat line: small lateral nudge only (large values erase odor).
OBSTACLE_SEEN_SOFT_BLEND = 0.14
# During threat, blend with *pure* follow_scent — not with soft bias (that double-counts avoidance).
# Lower b => more odor in the mix. Still ramp up avoidance when grass is very close (high in ROI).
AVOID_ODOR_BLEND_AT_BOUNDARY = 0.36
AVOID_ODOR_BLEND_WHEN_CLOSE = 0.88
# Head-on: both eyes see similar row — break tie with wider green band per eye.
BINOCULAR_HEIGHT_TIE_PX = 6

MAX_GRASS_WIDTH = 55           # I.e this would be the ground
NO_OBSTACLE_FOUND = -1
STEP_DRAGON_BOOST = 1000  # steps with boosted locomotion speed after detection
DRAGON_SPEED_FACTOR = 1.0  # multiply locomotion drive during boost (sensor rate unchanged)
DRAGON_FLY_DETECTION_THRESHOLD = 100 # threshold to detect when dragonfly head turn fully red, can be tuned based on the vision observation of the dragonfly head
SHOW_DRAGONFLY_DETECTION = False # set to True to visualize the dragonfly head detection process, which can help to tune STEP_DRAGON_BOOST and DRAGON_FLY_DETECTION_THRESHOLD
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
        self.alpha = 0.0075 * SENSOR_FREQ  # Smoothing factor for olfaction (adjusted for SENSOR_FREQ)

        self.avoid_hold = 0
        self.avoid_drive = np.array([1.5, 0.2])
        self.follow_drive = np.array([2.0, 2.0])
        self.follow_pure = np.array([2.0, 2.0])
        self._last_y_top = 0
        self._last_roi_h = 1
        self._seen_soft_active = False
        self._seen_soft_vec = np.array([2.0, 2.0])
        self._dragon_boost_remaining = 0
        self.current_step = 0
        self.print_freq = PRINT_FREQ

    def step(self, sim: MiniprojectSimulation, step):
        if step >45000: 
            self.print_freq = 2000
        self.current_step = step
        self.show_prints = SHOW_PRINTS and step % self.print_freq == 0

        raw_olfaction = sim.get_olfaction(sim.fly.name)
        if self.odor_smooth is None:
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (
                (1 - self.alpha) * self.odor_smooth
                + self.alpha * raw_olfaction
            )

        if step % SENSOR_FREQ == 0:
            (
                left_h,
                right_h,
                obstacle_threat,
                obstacle_seen,
                y_top,
                roi_h,
                left_w,
                right_w,
            ) = self.detect_obstacle(sim, visualize=self.show_prints)
            

            if self.show_prints:
                print(f"Vision obstacle scores: L={left_h:.4f}, R={right_h:.4f}")

            self._last_y_top = y_top
            self._last_roi_h = roi_h

            if obstacle_seen and self._olfaction_straight() and not obstacle_threat:
                wide_limit = max(int(roi_h * OBSTACLE_THREAT_STRAIGHT_FRAC), 1)
                if y_top <= wide_limit:
                    obstacle_threat = True

            if obstacle_seen and not obstacle_threat:
                self._seen_soft_active = True
                self._seen_soft_vec = self.avoid_obstacle(
                    left_h, right_h, left_w, right_w
                )
            else:
                self._seen_soft_active = False

            if obstacle_threat:
                self.avoid_drive = self.avoid_obstacle(left_h, right_h, left_w, right_w)
                self.avoid_hold = AVOID_HOLD_STEPS
            left_eye_score, right_eye_score, head_detected = self.detect_dragonflyhead(
                sim, step
            )
            if self.show_prints and step % self.print_freq == 0:
                print(
                    f"Dragonfly head scores: Left={left_eye_score}, "
                    f"Right={right_eye_score}, Detected={head_detected}"
                )

            if left_eye_score > right_eye_score:
                dragon_detected = left_eye_score > DRAGON_FLY_DETECTION_THRESHOLD
            else:
                dragon_detected = right_eye_score > DRAGON_FLY_DETECTION_THRESHOLD

            if dragon_detected:
                if self.show_prints:
                    print(
                        "Dragonfly head detected! "
                        f"Speed x{DRAGON_SPEED_FACTOR} for {STEP_DRAGON_BOOST} steps."
                    )
                self._dragon_boost_remaining = STEP_DRAGON_BOOST

        self.follow_pure = self.follow_scent()
        s_soft = OBSTACLE_SEEN_SOFT_BLEND if self._seen_soft_active else 0.0
        self.follow_drive = (1.0 - s_soft) * self.follow_pure + s_soft * self._seen_soft_vec

        if self.avoid_hold > 0:
            self.avoid_hold -= 1

        if self.avoid_hold > 0:
            self.state = State.AVOID_OBSTACLE
            b = self.closeness
            base_drive = b * self.avoid_drive + (1.0 - b) * self.follow_pure
            if self.show_prints and step % self.print_freq == 0:
                print(
                    f"Avoid_drive: {self.avoid_drive}, follow_pure: {self.follow_pure}, "
                    f"blend: {b:.2f}, resulting drive: {base_drive}"
                )
        else:
            self.state = State.FOLLOW_SCENT
            base_drive = self.follow_drive

        if self._dragon_boost_remaining > 0:
            self.state = State.DODGE_DRAGON
            self.drive = DRAGON_SPEED_FACTOR * base_drive
            self._dragon_boost_remaining -= 1
        else:
            self.drive = base_drive
        

        if step > 0 and self.show_prints:
            fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
            plt.figure(figsize=(10, 4))
            plt.imshow(fly_vision)
            plt.title(f"Vision step {step} — {self.state.name}")
            plt.axis("off")
            plt.show()

        joint_angles, adhesion = self.turning_controller.step(self.drive)

        return joint_angles, adhesion

    def detect_obstacle(
        self, sim, visualize=False, save_path=None, eye_titles=None, english=False
    ):
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
                end_w = int(w * (1 - BINOCULAR_OVERLAP_RATIO)) # we blind the fly's left eye in the center, to avoid it looking at the same region as the right eye
            else:       #right eye
                start_w = 0
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

            obstacle_y = NO_OBSTACLE_FOUND
            obstacle_width = 0

            for y in range(0, target_h, SWEEP_HEIGHT): #sweep the mask from top to bottom with a step of SWEEP_HEIGHT
                #This loop searches for a continous band of horizontal pixels, and consider it as grass if 
                # its width is between MIN_GRASS_WIDTH and MAX_GRASS_WIDTH
                band = green_mask[y:min(y + SWEEP_HEIGHT, target_h), :]
                
                col_has_green = np.any(band, axis=0).astype(int)
                
                diffs = np.diff(np.pad(col_has_green, (1, 1), 'constant'))
                
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                
                if len(starts) > 0:
                    max_continuous_width = np.max(ends - starts)
                else:
                    max_continuous_width = 0

                if MIN_GRASS_WIDTH <= max_continuous_width <= MAX_GRASS_WIDTH:
                    obstacle_y = y
                    obstacle_width = max_continuous_width
                    break


            heights.append(obstacle_y)
            widths.append(obstacle_width)
            
            if visualize:
                debug_info['imgs'].append(img.astype(np.uint8))
                debug_info['masks'].append(green_mask)
                debug_info['boxes'].append((0, start_w, target_h, end_w - start_w)) # y, x, h, w

        obstacle_seen = any(h != NO_OBSTACLE_FOUND for h in heights)
        h0 = int(np.asarray(eye_imgs[0]).shape[0])
        roi_h = max(int(h0 * SKY_REGION_RATIO), 1)
        y_top = roi_h
        left_w = widths[0] if len(widths) > 0 else 0
        right_w = widths[1] if len(widths) > 1 else 0

        obstacle_threat = False
        if obstacle_seen:
            y_top = min(h for h in heights if h != -1)
            limit = max(int(roi_h * OBSTACLE_THREAT_FRAC_OF_ROI), 1)
            obstacle_threat = y_top <= limit
            
            if self.show_prints:
                print(f"Eye {i}: obstacle_y={obstacle_y}, obstacle_width={obstacle_width}")
                print("ends - starts", ends - starts)

        if visualize and obstacle_seen:
            self.visualize_detection(
                debug_info,
                heights,
                save_path=save_path,
                eye_titles=eye_titles,
                english=english,
            )

        return (
            heights[0],
            heights[1],
            obstacle_threat,
            obstacle_seen,
            y_top,
            roi_h,
            left_w,
            right_w,
        )
    def detect_dragonflyhead(self,sim,steps):
        right_eye = False
        left_eye = False
        left_eye_score = 0
        right_eye_score = 0
        eye_imgs = sim.get_raw_vision(sim.fly.name)
        head_detected = False
        count = 0
        for img in eye_imgs:
            img = np.asarray(img)

            if img.max() <= 1.0:
                img = img * 255.0

            h, w, c = img.shape

            target_region = img

            r = target_region[:, :, 0].astype(float)
            g = target_region[:, :, 1].astype(float)
            b = target_region[:, :, 2].astype(float)

            # Simple color-based detection for the dragonfly's head 
            head_mask = (
                (r > 90) & (g < 30) & (b < 30)  # looking for a reddish color
            )
            if count == 1:
                right_eye = True
                right_eye_score = np.sum(head_mask)
                left_eye = False
            else:
                right_eye = False
                left_eye_score = np.sum(head_mask)
                left_eye = True

            if self.show_prints and SHOW_DRAGONFLY_DETECTION:
                if steps % self.print_freq == 0:
                    print(np.shape(head_mask))
                    print(f"right_eye_score: {right_eye_score}, left_eye_score: {left_eye_score}")
                    eye_label = "Left" if left_eye else "Right"
                    print(f"Dragonfly head detected: {np.any(head_mask)} ({eye_label} eye)")

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(target_region.astype(np.uint8))
                    axes[0].set_title(f"Dragonfly Vision ({eye_label} Eye)")
                    axes[0].axis('off')

                    axes[1].imshow(head_mask, cmap='gray')
                    axes[1].set_title("Dragonfly Head Mask (Reddish Detection)")
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.show()
                count += 1

            if np.any(head_mask):
                head_detected = True
                break

        return left_eye_score, right_eye_score, head_detected

    

    def visualize_detection(
        self, debug_info, heights, save_path=None, eye_titles=None, english=False
    ):
        """
        Affiche 4 graphiques : Les 2 images originales avec le cadre rouge,
        et les 2 masques verts avec une ligne/bande bleue indiquant la détection.
        If save_path is set, writes the figure to disk instead of plt.show().
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        if english:
            titles = eye_titles or ["Left eye", "Right eye"]
            title_raw = "compound eye (sky ROI)"
            title_mask = "green mask (sky ROI)"
            label_detected = "Detected blade top (y={y})"
            text_none = "No blade detected"
        else:
            titles = eye_titles or ["Œil Gauche", "Œil Droit"]
            title_raw = "Image Originale"
            title_mask = "Masque Vert (Rogné)"
            label_detected = "Haut détecté (Y={y})"
            text_none = "Rien détecté"

        for i in range(2):
            img = debug_info['imgs'][i]
            mask = debug_info['masks'][i]
            box_y, box_x, box_h, box_w = debug_info['boxes'][i]
            detected_y = heights[i]

            ax_img = axes[0, i]
            ax_img.imshow(img)
            ax_img.set_title(
                f"Step {self.current_step} — {titles[i]} — {title_raw}"
            )

            rect = patches.Rectangle((box_x, box_y), box_w, box_h, 
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax_img.add_patch(rect)
            
            ax_mask = axes[1, i]
            ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(
                f"Step {self.current_step} — {titles[i]} — {title_mask}"
            )
            
            if detected_y != NO_OBSTACLE_FOUND:
                rect_band = patches.Rectangle((0, detected_y), box_w, SWEEP_HEIGHT, 
                                              linewidth=0, facecolor='blue', alpha=0.4)
                ax_mask.add_patch(rect_band)
                
                ax_mask.axhline(detected_y, color='blue', linestyle='-', linewidth=2, 
                                label=label_detected.format(y=detected_y))
                
                ax_img.hlines(y=detected_y, xmin=box_x, xmax=box_x+box_w, colors='blue', linewidth=2)
                
                ax_mask.legend(loc="lower right")
            else:
                ax_mask.text(box_w/2, box_h/2, text_none, color='red', 
                             ha='center', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path:
            from pathlib import Path

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def avoid_obstacle(self, left_h, right_h, left_w=0.0, right_w=0.0):
        """Turn away from grass. Uses both eyes when possible; handles single-eye detection."""
        MISSING = -1
        left_ok = left_h != MISSING
        right_ok = right_h != MISSING

        if self.show_prints:
            print(f"AVOID_OBSTACLE: L={left_h:.4f}, R={right_h:.4f}")

        if left_ok and not right_ok:
            turn_right = True
            if self.show_prints:
                print("Obstacle côté œil gauche → tourner à droite")
        elif right_ok and not left_ok:
            turn_right = False
            if self.show_prints:
                print("Obstacle côté œil droit → tourner à gauche")
        elif left_ok and right_ok:
            dh = left_h - right_h
            if abs(dh) <= BINOCULAR_HEIGHT_TIE_PX:
                # Head-on: same row in both eyes — height comparison is arbitrary; use width.
                dw = float(left_w) - float(right_w)
                if abs(dw) > 2.0:
                    turn_right = dw > 0.0
                elif abs(dh) < 1e-6:
                    # Perfect head-on symmetry; dh < 0 was always False — pick a fixed default.
                    turn_right = True
                else:
                    turn_right = dh < 0.0
            else:
                turn_right = left_h < right_h
            if self.show_prints:
                print(
                    "Obstacle binocular → tourner à "
                    + ("droite" if turn_right else "gauche")
                )
        else:
            turn_right = True

        ys = [h for h in (left_h, right_h) if h != MISSING]
        y_near = min(ys) if ys else OBSTACLE_ROW_CLOSE
        if y_near < OBSTACLE_ROW_CLOSE:
            fast, slow = 1.35, 0.06
            self.closeness = 1.0
            if self.show_prints:
                print("Obstacle très proche → évitement serré mais toujours en avant")
        elif y_near < OBSTACLE_ROW_CLOSE * 2:
            fast, slow = 1.10, 0.08
            self.closeness = 0.9
            if self.show_prints:
                print("Obstacle proche → évitement modéré avec moins d'avance")
        else:
            fast, slow = 0.95, 0.10
            self.closeness = 0.2
            if self.show_prints:
                print("Obstacle à distance → évitement doux et lent")

        if turn_right:
            return np.array([fast, slow])
        return np.array([slow, fast])

    def _olfaction_straight(self):
        """True when left/right odor balance matches 'go straight' in follow_scent."""
        left_odor_a = self.odor_smooth[0, 0]
        left_odor_b = self.odor_smooth[2, 0]
        right_odor_a = self.odor_smooth[1, 0]
        right_odor_b = self.odor_smooth[3, 0]
        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b
        eps = 1e-8
        ratio = abs(left_signal) / (abs(right_signal) + eps)
        return abs(ratio - 1.0) < GO_STRAIGHT_THRESHOLD

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
            # print(f"Olfaction: {self.odor_smooth}")
            # print(f"Left signal: {left_signal}, Right signal: {right_signal}")
            print(f"Olfaction ratio: {ratio}")

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