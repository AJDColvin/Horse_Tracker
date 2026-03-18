import cv2
import math
import torch
from collections import deque
from ultralytics import YOLO
import csv

# MOVEMENT_THRESHOLD = 600.0
# MIN_STATE_TIME = 2.0
# INDIVIDUALS = 2
FPS = 15
WINDOW_SIZE = 15
STRIDE = 2

id_to_replace = list(range(14,24))
id_to_replace.append(0)

class HorseTracker:
        def __init__(
            self, 
            model_path: str, 
            video_path:str, 
            csv_path:str = "horse_activity_log.csv", 
            save_path:str = None, 
            custom_model: bool = False,
            movement_threshold: float = 600.0,
            min_state_time: float = 2.0,
            individuals: int = 2
        ):
            
            # Filepaths
            self.model = YOLO(model_path)
            self.video_path = video_path
            self.csv_path = csv_path
            self.save_path = save_path
            
            # Constants
            self.movement_threshold = movement_threshold
            self.min_state_time = min_state_time
            self.individuals = individuals
            
            # Initialize state variables
            self.windows = [deque(maxlen=WINDOW_SIZE) for _ in range(individuals)]
            self.prev_coordinates = {}
            self.horse_states = {i: "OUT_OF_FRAME" for i in range(individuals)}
            self.state_history = {i: [] for i in range(individuals)}
            self.valid_ids = list(range(1, individuals + 1))
            self.frame_data = {} # Cache bounding box data for ammended video
            
            self.custom_model = custom_model
        
         
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
            if not self.custom_model: 
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
            
            if window_avg > self.movement_threshold:
                current_state = "MOVING"
            else:
                current_state = "STILL"
            
            return current_state

        def _draw_annotations(self, frame, x1, y1, x2, y2, state, amending=False, horse_id: int = None):
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
                if amending:
                    cv2.putText(    
                        frame,
                        f"ID {horse_id + 1}",
                        (int(x1), int(y1) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0,255,0),
                        thickness=2
                    )
            else:
                cv2.rectangle(
                    frame,
                    (int(x1),int(y1)),
                    (int(x2),int(y2)),
                    (147,20,255),
                    2
                )
                cv2.putText(
                    frame,
                    'STILL',
                    (int(x1),int(y2)+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (147,20,255),
                    thickness= 2
                )
                if amending:
                    cv2.putText(    
                        frame,
                        f"ID {horse_id + 1}",
                        (int(x1), int(y1) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (147,20,255),
                        thickness=2
                    )
        
        def _update_state_history(self, id_val: int, current_state: str, total_seconds: float):
            previous_state = self.horse_states[id_val]
            
            if current_state != previous_state:
                if self.state_history[id_val]:
                    state_time = total_seconds - self.state_history[id_val][-1]['timestamp']
                else:
                    state_time = total_seconds
                    
                if state_time > self.min_state_time:
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

                # Final timestam
        
        def _export_csv(self, filename: str ="horse_activity_log.csv"):
            
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)

                for horse_id, history in self.state_history.items():
                    
                    # Headings 
                    writer.writerow([f"Horse {horse_id + 1}", "", ""])
                    writer.writerow(["State", "Start", "End"])
                    
                    total_moving_seconds = 0.0 # Total time spent moving
                    total_duration = 0.0
                    start_time = 0.0
                    
                    # Main data, all timestamps
                    for record in history:
                        state = record['changed_from']
                        end_time = record['timestamp']
                        
                        start_str = self._format_timestamps(start_time)
                        end_str = self._format_timestamps(end_time)
                        
                        writer.writerow([state, start_str, end_str])
                        
                        duration = end_time - start_time
                        if state == "MOVING":
                            total_moving_seconds += duration
                            
                        start_time = end_time
                        total_duration = end_time
                        
                    # Final row with overall stats
                    writer.writerow(["", "Total Activity", "Perc of Day, Moving"])
        
                    moving_str = self._format_timestamps(total_moving_seconds)
                    
                    if total_duration > 0:
                        percent_moving = (total_moving_seconds/ total_duration) * 100
                    else:
                        percent_moving = 0.0 
                      
                    percent_str = f"{percent_moving:.1f}%"
                    writer.writerow(["", moving_str, percent_str])
                    
                    writer.writerow([])
           
        def _get_smoothed_state(self, horse_id: int, current_time: float) -> str:
            """Determines the smoothed state of horse at a specific timestamp"""
            
            history = self.state_history.get(horse_id, [])
            
            for record in history:
                if current_time <= record['timestamp']:
                    return record['changed_from']
            
            return self.horse_states.get(horse_id, "OUT_OF_FRAME")   
        
        def _save_amended_video(self):
            """Creates a new video with smoothed states applied."""    
            cap = cv2.VideoCapture(self.video_path)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # * parses 'm''p' '4' 'v', which is the format expected
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            output_fps = original_fps / STRIDE 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(self.save_path, fourcc, output_fps, (width, height))
            
            frame_idx = 0      
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames to match the YOLO vid_stride
                if frame_idx % STRIDE != 0:
                    frame_idx += 1
                    continue
                
                processed_frame_no = frame_idx // STRIDE
                total_seconds = processed_frame_no * STRIDE / FPS
                
                frame_boxes = self.frame_data.get(processed_frame_no, {})
                
                for horse_id, box in frame_boxes.items():
                    print("INSIDE SAVEAMENDEDVIDEO: ", end=" ")
                    print(horse_id)
                    smoothed_state = self._get_smoothed_state(horse_id, total_seconds)
                    x1, y1, x2, y2 = box
                    self._draw_annotations(frame, x1, y1, x2, y2, smoothed_state, amending=True, horse_id=horse_id)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            print(f"Saved to {self.filename}")
                    
                     
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
                    
                    # Rectify all animals to horse, unless only 1 class (in custom model)
                    id_to_box, annotated_frame = self._rectify_ids(result)
                    
                    for id_val in range(self.individuals):
                        box = id_to_box.get(id_val+1)
                        current_state = "OUT_OF_FRAME"
                        
                        if box is not None:
                            current_state = self._calculate_movement(id_val, box, frame_no)
                            
                            x1, y1, x2, y2 = box
                            
                            # Draw annotations on video
                            self._draw_annotations(annotated_frame, x1, y1, x2, y2, current_state)
                            
                            # Cache bounding boxes for ammended video
                            if frame_no not in self.frame_data:
                                self.frame_data[frame_no] = {}
                            self.frame_data[frame_no][id_val] = (x1, y1, x2, y2)  
                        
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
            
            # Final timestamps logic
            for horse_id in range(self.individuals):
                    self.state_history[horse_id].append({
                        "timestamp": total_seconds,
                        "changed_from": self.horse_states[horse_id]
                    })
                
            self._print_summary(total_seconds)
            self._export_csv()
            if self.save_path:
                self._save_amended_video()
            
if __name__ == "__main__":
    MODEL_PATH = '../YOLO_models/yolo11s_Professor_M_Horses-2_F10.pt'
    VIDEO_PATH = '/Volumes/USB Drive/TAPO_clips/2_individuals_1_leave_return.mp4.mov'
    
    tracker = HorseTracker(MODEL_PATH, VIDEO_PATH, custom_model=True)
    tracker.run()       
            
    
