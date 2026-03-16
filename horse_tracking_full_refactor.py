import cv2
import math
import torch
from collections import deque
from ultralytics import YOLO

MOVEMENT_THRESHOLD = 600.0
MIN_STATE_TIME = 2.0
INDIVIDUALS = 2
FPS = 15
WINDOW_SIZE = 15
STRIDE = 2

id_to_replace = list(range(14,24))
id_to_replace.append(0)

class HorseTracker:
        def __init__(self, model_path: str, video_path:str):
            self.model = YOLO(model_path)
            self.video_path = video_path
            
            # Initialize state variables
            self.windows = [deque(maxlen=WINDOW_SIZE) for _ in range(INDIVIDUALS)]
            self.prev_coordinates = {}
            self.horse_states = {i: "OUT_OF_FRAME" for i in range(INDIVIDUALS)}
            self.state_history = {i: [] for i in range(INDIVIDUALS)}
            self.valid_ids = list(range(1, INDIVIDUALS + 1))
        
         
        # Internal Methods   
        def _format_timestamps(self, total_seconds: float) -> str:
            """Helper method to format seconds into MM:SS.mmm"""
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds % 1) * 1000)
            return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}" 
        
        def _rectify_ids(self, result) -> dict:
            """Handles logic for overriding animal classes with horse class"""
            
            # Replace any animal detections with horses
            amended_boxes = result.boxes.data.clone()
            
            for class_id in id_to_replace:
                amended_boxes[:, -1][amended_boxes[:, -1] == class_id] = 17
            
            # annotated_frame = result.plot(line_width=2)
            # boxes = result.boxes.xyxy.cpu().numpy()
            
            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy() 
        
                # Replace invalid IDs with valid ones 
                valid_ids_oof = list(set(self.valid_ids)-set(ids)) # Find valid ids, not in current ids
                replacement_ids = iter(valid_ids_oof)
                set_valid_ids = set(self.valid_ids)
                rectified_ids = [next(replacement_ids, x) if x not in set_valid_ids else x for x in ids]
                
                new_ids_tensor = torch.tensor(rectified_ids, device=amended_boxes.device, dtype=amended_boxes.dtype)

                # 4. Overwrite the ID column (Index 4)
                amended_boxes[:, 4] = new_ids_tensor

                result.boxes.data = amended_boxes
                
                annotated_frame = result.plot(line_width=2)
                
                
                ids = result.boxes.id.cpu().numpy() 
                boxes = result.boxes.xyxy.cpu().numpy()
                id_to_box = dict(zip(ids.astype(int), boxes))
            
            else:
                result.boxes.data = amended_boxes
                annotated_frame = result.plot(line_width=2)
                id_to_box = {} # If no ids at all, every individual given OOF           
        
            return id_to_box, annotated_frame
            
        def _calculate_movement(self, id_val: int, box: tuple, frame_no: int) -> str:
            """Calculates centroid movement and returns current state"""
            x1, y1, x2, y2 = box
            box_area = (x2-x1)*(y2-y1)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Euclidian distance
            if frame_no == 0 or id_val not in self.prev_coordinates:
                c_distance = 0
            else:
                prev_cx, prev_cy = self.prev_coordinates[id_val]
                c_distance = math.dist((cx,cy),(prev_cx,prev_cy))
                
            self.prev_coordinates[id_val] = (cx, cy)
                            
            c_distance_normalised = (c_distance/math.cbrt(box_area))*10000
            self.windows[id_val].append(c_distance_normalised)
            
            window_avg = sum(self.windows[id_val])/len(self.windows[id_val])
            
            if window_avg > MOVEMENT_THRESHOLD:
                current_state = "MOVING"
            else:
                current_state = "STILL"
            
            return current_state

        def _draw_annotations(self, frame, x1, y1, x2, y2, state):
            """Draws boxes and text on the current frame"""
            
            if state == "MOVING":
                cv2.rectangle(
                    frame,
                    (int(x1),int(y1)),
                    (int(x2),int(y2)),
                    (0,255,0),
                    2
                )
                cv2.putText(
                    frame,
                    'MOVE',
                    (int(x1),int(y2)+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0,255,0),
                    thickness= 2
                )
            else:
                cv2.putText(
                    frame,
                    'STILL',
                    (int(x1),int(y2)+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (147,20,255),
                    thickness= 2
                )
        
        def _update_state_history(self, id_val: int, current_state: str, total_seconds: float):
            previous_state = self.horse_states[id_val]
            
            if current_state != previous_state:
                if self.state_history[id_val]:
                    state_time = total_seconds - self.state_history[id_val][-1]['timestamp']
                else:
                    state_time = total_seconds
                    
                if state_time > MIN_STATE_TIME:
                    self.state_history[id_val].append({
                        "timestamp": total_seconds,
                        "changed_from": previous_state
                    })
                    
                    # Update the active state record
                    self.horse_states[id_val] = current_state
        
        def _print_summary(self, total_seconds):
            print("\n--- FINAL STATE HISTORY ---")
            for horse_id, history in self.state_history.items():
                print(f"\nHorse {horse_id + 1}:")
                if not history:
                    print("  No state changes recorded.")
                for record in history:
                    minutes = int(record['timestamp'] // 60)
                    seconds = int(record['timestamp'] % 60)
                    milliseconds = int((record['timestamp'] % 1) * 1000)
                    timestamp = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                    print(f"  [{timestamp}] to {record['changed_from']}")

                # Final timestamp
                timestamp = self._format_timestamps(total_seconds)
                print(f"  [{timestamp}] to {self.horse_states[horse_id]}")
        
        def _export_csv(self, filepath="./"):
            pass

            
        def run(self):
            """Main execution loop"""
            results = self.model.track(
                tracker="custom_botsort.yaml",
                source=self.video_path,
                show=False,
                stream=True, 
                conf=0.1,
                device='mps',
                agnostic_nms=True,
                verbose=False, 
                vid_stride=STRIDE,
                imgsz=1080
            )   
            
            for frame_no, result in enumerate(results):
                # OVERALL MAIN LOOP
                total_seconds = frame_no * STRIDE / FPS
                
                if result.boxes is not None:
                    
                    id_to_box, annotated_frame = self._rectify_ids(result)
                    
                    for id_val in range(INDIVIDUALS):
                        box = id_to_box.get(id_val+1)
                        current_state = "OUT_OF_FRAME"
                        
                        if box is not None:
                            current_state = self._calculate_movement(id_val, box, frame_no)
                            
                            x1, y1, x2, y2 = box
                            self._draw_annotations(annotated_frame, x1, y1, x2, y2, current_state)
                        
                        else:
                            self.windows[id_val].clear()
                            if id_val in self.prev_coordinates:
                                del self.prev_coordinates[id_val]
                                
                        print(f"HORSE {id_val+1}: {current_state}", end=" ")
                        
                        self._update_state_history(id_val, current_state, total_seconds)
                    print('')
                else:
                    annotated_frame = result.orig_img.copy()
                    
                cv2.imshow("Horse Tracking", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
            self._print_summary(total_seconds)
            self._export_csv()
            
if __name__ == "__main__":
    MODEL_PATH = '../YOLO_models/yolo11s.pt'
    VIDEO_PATH = '/Volumes/USB Drive/TAPO_clips/2_individuals_1_leave_return.mp4.mov'
    
    tracker = HorseTracker(MODEL_PATH, VIDEO_PATH)
    tracker.run()       
            
    
