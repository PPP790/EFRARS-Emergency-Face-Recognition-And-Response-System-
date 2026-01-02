import cv2
import os
import pandas as pd
import numpy as np
import time
import pickle
import threading
import queue
import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Enum, func
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.exc import IntegrityError
from deepface import DeepFace
from scipy.spatial import distance
from datetime import datetime
from twilio.rest import Client
from flask import Flask, render_template, Response, jsonify

# --- Configuration (Final Balanced CPU Stack) ---
EMPLOYEE_CSV_PATH = 'database.csv'
IMAGE_BASE_DIR = 'employee_images'
DATABASE_FILE = 'emergency_system.db'
MODEL_NAME = 'ArcFace'              # Use ArcFace (matches cosine logic)
DETECTOR_BACKEND = 'ssd'            # Use 'ssd' for the best CPU speed/accuracy balance
DISTANCE_THRESHOLD = 0.6          # A good starting threshold for ArcFace/Cosine. Tune this.
FRAME_SKIP = 5                      # Process every 5th frame to reduce processing load

# --- SECURITY: Load sensitive credentials from Environment Variables ---
# --- SECURITY: Load sensitive credentials from Environment Variables ---
# Make sure to set these in your terminal before running
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER')

if not (os.environ.get('TWILIO_ACCOUNT_SID') and os.environ.get('TWILIO_AUTH_TOKEN')):
     print("--- WARNING: Twilio credentials not fully set in environment variables. Using hardcoded fallbacks. ---")
     print("--- This is INSECURE. Please set these as environment variables. ---")
# --- END SECURITY ---

# --- Flask App Setup ---
app = Flask(__name__)

# --- Database Setup using SQLAlchemy ---
Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    phone = Column(String)
    status = Column(Enum('Unaccounted', 'Safe', name='status_enum'), default='Unaccounted', nullable=False)
    face_embedding = Column(Text, nullable=False) # This will store the PICKLED MASTER embedding
    image_hash = Column(String)

class Detection(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    camera_id = Column(String, default='webcam_prototype')

engine = create_engine(f'sqlite:///{DATABASE_FILE}')
Session = sessionmaker(bind=engine)
db_session_factory = Session
db_session = scoped_session(db_session_factory)

# --- Global variables for camera/processing threads ---
input_frame_queue = queue.Queue(maxsize=10)
output_frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()
processing_active = False
camera_thread = None
processing_thread = None

# --- Helper function for hashing directory contents (Unchanged) ---
def calculate_directory_hash(directory_path):
    """Calculates an SHA256 hash of a directory's contents."""
    hasher = hashlib.sha256()
    if not os.path.exists(directory_path):
        return None
    
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        
        for name in files:
            filepath = os.path.join(root, name)
            hasher.update(name.encode('utf-8'))
            hasher.update(str(os.path.getsize(filepath)).encode('utf-8'))
            
            try:
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
            except IOError:
                pass
    return hasher.hexdigest()

# --- Core Functions ---
def setup_database():
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        print(f"Error setting up database: {e}")
        exit()

def reset_employee_statuses():
    session = db_session()
    try:
        session.query(Detection).delete()
        all_employees = session.query(Employee).all()
        for emp in all_employees:
            emp.status = 'Unaccounted'
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error resetting employee statuses: {e}")
    finally:
        session.close()
        db_session.remove()

def enroll_employees():
    """
    --- REVERTED ENROLLMENT FUNCTION (v5) ---
    This version now creates a single "master embedding" by averaging all
    high-confidence photos for a person.
    """
    session = db_session()
    try:
        # Check for GPU availability
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"--- Found {len(gpus)} GPU(s). TensorFlow will use GPU. ---")
            else:
                print("--- No GPU found. TensorFlow will use CPU (Enrollment may be slow). ---")
        except ImportError:
            pass
        except Exception as e:
            print(f"An error occurred during GPU check: {e}")

        _ = DeepFace.build_model(model_name=MODEL_NAME)
        print(f"--- Successfully built DeepFace model '{MODEL_NAME}' ---")
    except Exception as e:
        print(f"Error loading DeepFace model '{MODEL_NAME}': {e}.")
        print("Please ensure you have a stable internet connection and necessary dependencies.")
        session.close()
        db_session.remove()
        return False

    if not os.path.exists(EMPLOYEE_CSV_PATH):
        print(f"Error: Employee CSV file not found at '{EMPLOYEE_CSV_PATH}'.")
        session.close()
        db_session.remove()
        return False

    try:
        df = pd.read_csv(EMPLOYEE_CSV_PATH)
        df.columns = df.columns.str.strip()
        if df.empty:
            session.close()
            db_session.remove()
            return False
    except Exception as e:
        print(f"Error reading CSV file '{EMPLOYEE_CSV_PATH}': {e}")
        session.close()
        db_session.remove()
        return False

    enrollment_successful = False
    for index, row in df.iterrows():
        if 'name' not in row or 'phone' not in row:
            print(f"ERROR: 'name' or 'phone' column not found in row. Skipping.")
            continue
        employee_name = str(row['name']).strip()
        employee_phone = str(row['phone']).strip()

        existing_employee = session.query(Employee).filter_by(name=employee_name).first()
        
        employee_image_dir = os.path.join(IMAGE_BASE_DIR, employee_name)
        if not os.path.isdir(employee_image_dir):
            print(f"Warning: Image directory '{employee_image_dir}' not found for '{employee_name}'. Skipping.")
            continue

        current_directory_hash = calculate_directory_hash(employee_image_dir)
        if current_directory_hash is None:
            print(f"Warning: Could not calculate hash for '{employee_name}' image directory. Skipping.")
            continue

        if existing_employee and existing_employee.image_hash == current_directory_hash:
            print(f"--- Employee '{employee_name}' already enrolled and images are unchanged. Skipping. ---")
            enrollment_successful = True
            continue

        print(f"--- Enrolling / Updating employee: {employee_name} ---")
        image_files = [os.path.join(employee_image_dir, f) for f in os.listdir(employee_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"Warning: No images found in '{employee_image_dir}' for '{employee_name}'. Skipping.")
            continue

        embeddings_list = [] 
        for img_path in image_files:
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True 
                )
                if not face_objs:
                    print(f"No face detected in {os.path.basename(img_path)}. Skipping.")
                    continue

                face_obj = face_objs[0] 
                
                if face_obj['confidence'] < 0.90:
                    print(f"Skipping {os.path.basename(img_path)}, detection confidence too low ({face_obj['confidence']:.2f}).")
                    continue

                embedding = DeepFace.represent(
                    img_path=face_obj['face'], 
                    model_name=MODEL_NAME,
                    enforce_detection=False 
                )[0]['embedding']
                
                embeddings_list.append(embedding)

            except Exception as e:
                print(f"Error processing image {os.path.basename(img_path)}: {e}")

        if not embeddings_list:
            print(f"Error: Could not generate any HIGH-CONFIDENCE embeddings for '{employee_name}'. Please check images.")
            continue

        try:
            # --- THIS IS THE REVERTED LOGIC ---
            # We are now averaging all the good embeddings into one.
            master_embedding = np.mean(embeddings_list, axis=0).tolist()
            embedding_str = pickle.dumps(master_embedding)
            # --- END REVERTED LOGIC ---
        except Exception as e:
            print(f"ERROR: Failed to average or serialize embedding for '{employee_name}': {e}")
            continue

        if existing_employee:
            existing_employee.face_embedding = embedding_str
            existing_employee.phone = employee_phone
            existing_employee.image_hash = current_directory_hash
            session.add(existing_employee)
        else:
            new_employee = Employee(
                name=employee_name,
                phone=employee_phone,
                status='Unaccounted',
                face_embedding=embedding_str,
                image_hash=current_directory_hash
            )
            session.add(new_employee)

        try:
            session.commit()
            enrollment_successful = True
        except IntegrityError:
            session.rollback()
            print(f"Error: An employee with name '{employee_name}' might already exist. Rolled back transaction.")
        except Exception as e:
            session.rollback()
            print(f"A database error occurred while enrolling {employee_name}: {e}. Rolled back transaction.")
            
    session.close()
    db_session.remove()
    return enrollment_successful


def _send_sms_alert_threaded(message_body):
    """Helper function to send SMS in a separate thread."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, RECIPIENT_PHONE_NUMBER]):
        print("\n--- SMS Alert FAILED: Twilio credentials are not set. ---")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            body=message_body
        )
        print(f"\n--- Successfully sent SMS Alert to {RECIPIENT_PHONE_NUMBER} ---")
    except Exception as e:
        print(f"\nError sending SMS alert: {e}")

def send_sms_alert(message_body):
    """Spawns a new thread to send an SMS alert."""
    alert_thread = threading.Thread(target=_send_sms_alert_threaded, args=(message_body,))
    alert_thread.daemon = True
    alert_thread.start()


class CameraCaptureThread(threading.Thread):
    """
    Thread for capturing frames from the camera and putting them into a queue.
    """
    def __init__(self, input_frame_queue, stop_event):
        super().__init__()
        self.input_frame_queue = input_frame_queue
        self.stop_event = stop_event
        self.cap = None
        self.name = "CameraCaptureThread"

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print(f"[{self.name}] Camera 0 not found, trying Camera 1...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print(f"[{self.name}] ERROR: Could not open any camera. Stopping capture.")
            self.stop_event.set()
            return

        print(f"[{self.name}] Successfully opened camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30) 

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.name}] Error: Failed to grab frame. Stopping capture.")
                self.stop_event.set()
                break
            try:
                self.input_frame_queue.put_nowait(frame)
            except queue.Full:
                pass 
            time.sleep(0.01) 
            
        self.cap.release()
        print(f"[{self.name}] Camera released.")


class FaceProcessingThread(threading.Thread):
    """
    --- THIS IS THE FULLY OPTIMIZED THREAD ---
    Uses the in-memory cache and 1-to-Many matching logic.
    """
    def __init__(self, input_frame_queue, output_frame_queue, stop_event, db_session_factory):
        super().__init__()
        self.input_frame_queue = input_frame_queue
        self.output_frame_queue = output_frame_queue
        self.stop_event = stop_event
        self.db_session_factory = db_session_factory
        self.name = "FaceProcessingThread"
        self.deepface_model_instance = None
        self.frame_counter = 0

        # --- CACHE LOGIC ---
        self.employee_data_cache = {}  
        self.known_embeddings_matrix = None 
        self.known_ids_list = []          
        # --- END CACHE ---

    def _load_employee_cache(self, session):
        """
        --- REVERTED CACHE LOADER ---
        This function now reads the single MASTER embedding for each person
        and loads it into the recognition matrix.
        """
        print(f"[{self.name}] Caching employee embeddings from database...")
        all_employees = session.query(Employee).all()
        
        embeddings_list = []      
        self.known_ids_list = []       
        self.employee_data_cache = {}  

        for emp in all_employees:
            try:
                # 1. Store the simple employee data (name, status) in the cache by ID
                self.employee_data_cache[emp.id] = {
                    'name': emp.name,
                    'status': emp.status
                }
                
                # 2. Load the data from the DB, which is now a single MASTER embedding
                master_embedding = pickle.loads(emp.face_embedding)
                
                # 3. Add this single embedding to our lists for the recognition matrix
                embeddings_list.append(np.array(master_embedding))
                self.known_ids_list.append(emp.id)

            except Exception as e:
                print(f"[{self.name}] FAILED to load/unpickle embedding for {emp.name}: {e}")
                print("--- This may be due to corrupt data in the database. Please RE-DELETE database and restart. ---")

        if embeddings_list:
            # 4. Create the final matrix of one master embedding per employee
            self.known_embeddings_matrix = np.array(embeddings_list)
            print(f"[{self.name}] Successfully cached {len(embeddings_list)} master embeddings for {len(self.employee_data_cache)} employees.")
        else:
            print(f"[{self.name}] WARNING: No employees found in database to cache.")
        
    def run(self):
        """The main processing loop for the thread."""
        session = self.db_session_factory()
        
        try:
            self._load_employee_cache(session) 
            if self.known_embeddings_matrix is None or len(self.known_ids_list) == 0:
                print(f"[{self.name}] No employee data cached. Cannot proceed. Stopping thread.")
                self.stop_event.set()
                session.close()
                db_session.remove()
                return
                
            self.deepface_model_instance = DeepFace.build_model(model_name=MODEL_NAME)
            print(f"[{self.name}] DeepFace model loaded and cache built. Starting processing loop.")
        except Exception as e:
            print(f"[{self.name}] CRITICAL ERROR on thread startup: {e}. Signalling stop.")
            self.stop_event.set()
            session.close()
            db_session.remove()
            return

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.input_frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue 

                self.frame_counter += 1
                process_frame = frame.copy()
                
                if self.frame_counter % FRAME_SKIP == 0:
                    try:
                        uncommitted_detections = [] 
                        
                        detected_faces = DeepFace.extract_faces(
                            img_path=process_frame,
                            detector_backend=DETECTOR_BACKEND,
                            enforce_detection=False
                        )
                        
                        for face_obj in detected_faces:
                            if face_obj['confidence'] == 0:  
                                continue

                            x, y, w, h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], face_obj['facial_area']['w'], face_obj['facial_area']['h']
                            
                            face_embedding = DeepFace.represent(
                                img_path=face_obj['face'], 
                                model_name=MODEL_NAME,
                                enforce_detection=False 
                            )[0]['embedding']
                            
                            face_embedding_arr = np.array(face_embedding)

                            distances = distance.cdist(np.expand_dims(face_embedding_arr, axis=0), self.known_embeddings_matrix, 'cosine')[0]
                            
                            best_match_index = np.argmin(distances)     
                            best_match_distance = distances[best_match_index] 
                            
                            # print(f"--- DEBUG: Best match distance = {best_match_distance:.4f} ---")

                            identified_name = "Unknown"
                            color = (0, 0, 255) 

                            if best_match_distance < DISTANCE_THRESHOLD:
                                matched_id = self.known_ids_list[best_match_index]
                                matched_employee_cache = self.employee_data_cache.get(matched_id)

                                if matched_employee_cache:
                                    identified_name = matched_employee_cache['name']
                                    color = (0, 255, 0) 

                                    uncommitted_detections.append(Detection(employee_id=matched_id, timestamp=datetime.now()))

                                    if matched_employee_cache['status'] == 'Unaccounted':
                                        print(f"[{self.name}] First-time detection: Marking {identified_name} as Safe.")
                                        matched_employee_cache['status'] = 'Safe' 
                                        employee_to_update = session.query(Employee).get(matched_id)
                                        if employee_to_update:
                                            employee_to_update.status = 'Safe'
                                            session.add(employee_to_update)
                            
                            cv2.rectangle(process_frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(process_frame, identified_name.title(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        if uncommitted_detections:
                            session.add_all(uncommitted_detections)
                            session.commit()

                    except Exception as e:
                        if "Face detector" in str(e) or "No face" in str(e) or "cannot reshape" in str(e):
                            pass 
                        else:
                            print(f"[{self.name}] ERROR during frame processing: {e}")
                        session.rollback() 

                try:
                    ret, buffer = cv2.imencode('.jpg', process_frame)
                    if ret:
                        self.output_frame_queue.put_nowait(buffer.tobytes())
                except queue.Full:
                    pass 
        
        finally:
            print(f"[{self.name}] Stopping loop. Closing session.")
            session.close()
            db_session.remove()


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            if not processing_active or stop_event.is_set():
                img = np.zeros((480, 640, 3), dtype=np.uint8) + 50
                cv2.putText(img, "Monitoring Inactive", (150, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', img)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue

            try:
                frame_bytes = output_frame_queue.get(timeout=1.0) 
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                pass 
            except Exception as e:
                print(f"Error in video_feed generation: {e}")
                break 

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_emergency', methods=['POST'])
def toggle_emergency():
    global processing_active, camera_thread, processing_thread

    if not processing_active:
        print("\n--- ACTION: STARTING Emergency Monitoring ---")
        with input_frame_queue.mutex:
            input_frame_queue.queue.clear()
        with output_frame_queue.mutex:
            output_frame_queue.queue.clear()
        
        stop_event.clear()
        reset_employee_statuses()

        camera_thread = CameraCaptureThread(input_frame_queue, stop_event)
        processing_thread = FaceProcessingThread(input_frame_queue, output_frame_queue, stop_event, db_session_factory)

        camera_thread.start()
        processing_thread.start()
        processing_active = True
        return jsonify(status="started", message="Emergency recognition started.")
    else:
        print("\n--- ACTION: STOPPING Emergency Monitoring ---")
        stop_event.set()
        
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=2)
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=2)

        print("--- Generating final report... ---")
        session = db_session()
        try:
            unaccounted_employees_list = session.query(Employee).filter_by(status='Unaccounted').all()
            if unaccounted_employees_list:
                alert_message = f"EMERGENCY REPORT: {len(unaccounted_employees_list)} personnel are UNACCOUNTED for:\n"
                for emp in unaccounted_employees_list:
                    alert_message += f"- {emp.name.title()} (Phone: {emp.phone})\n"
                send_sms_alert(alert_message.strip())
            else:
                 alert_message = "EMERGENCY REPORT: All personnel are accounted for and marked SAFE."
                 send_sms_alert(alert_message)
        finally:
            session.close()
            db_session.remove()

        processing_active = False
        return jsonify(status="stopped", message="Emergency recognition stopped. Final report sent.")

@app.route('/status_update')
def status_update():
    session = db_session()
    try:
        safe_employees_data = []
        unaccounted_employees_data = []

        latest_detection_subquery = session.query(
            Detection.employee_id,
            func.max(Detection.timestamp).label('latest_timestamp')
        ).group_by(Detection.employee_id).subquery()

        safe_employees_query = session.query(
            Employee,
            latest_detection_subquery.c.latest_timestamp
        ).join(
            latest_detection_subquery, Employee.id == latest_detection_subquery.c.employee_id
        ).filter(
            Employee.status == 'Safe'
        ).order_by(
            latest_detection_subquery.c.latest_timestamp.desc()
        ).all()

        for emp, timestamp in safe_employees_query:
            safe_employees_data.append({
                'id': emp.id,
                'name': emp.name.title(),
                'phone': emp.phone,
                'detectedAt': timestamp.strftime('%H:%M:%S')
            })

        unaccounted_employees = session.query(Employee).filter_by(status='Unaccounted').all()
        for emp in unaccounted_employees:
            unaccounted_employees_data.append({
                'id': emp.id,
                'name': emp.name.title(),
                'phone': emp.phone
            })
        
        return jsonify({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'safe': safe_employees_data,
            'unaccounted': unaccounted_employees_data,
            'processing_active': processing_active 
        })
    except Exception as e:
        print(f"Error fetching status update: {e}")
        session.rollback()
        return jsonify(error="Failed to fetch status", details=str(e)), 500
    finally:
        session.close()
        db_session.remove()

if __name__ == '__main__':
    setup_database()
    print("\n--- Running Employee Enrollment Process ---")
    enroll_employees()
    print("--- Employee Enrollment Process Finished ---")
    
    temp_session = db_session()
    total_enrolled = temp_session.query(Employee).count()
    temp_session.close()
    db_session.remove()

    if total_enrolled == 0:
        print("\n--- CRITICAL: No employees were enrolled. ---")
        print("--- Please ensure 'database.csv' exists and 'employee_images' folder is populated. ---")
        print("--- Exiting program. ---")
        exit()
    else:
        print(f"\n--- System Ready. Found {total_enrolled} enrolled employees. ---")
        print(f"--- Starting Flask server at http://0.0.0.0:5000 ---")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
