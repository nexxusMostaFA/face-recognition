from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pymongo
from bson.binary import Binary
import pickle
import time
import uuid
import logging
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FaceRecognitionAPI')

class FaceRecognitionAPI:
    def __init__(self, mongodb_uri, db_name, collection_name):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
         
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
         
        self.initialize_model()
    
        self.upload_folder = 'uploads'
        os.makedirs(self.upload_folder, exist_ok=True)
        
    def initialize_model(self):
        logger.info("Downloading and initializing AuraFace model...")
        try:
            snapshot_download(
                "fal/AuraFace-v1",
                local_dir="models/auraface",
            )
            
         
            self.face_app = FaceAnalysis(
                name="auraface",
                providers=["CPUExecutionProvider"],
                root=".",
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def process_image(self, image_path):
        """Process an image and detect faces"""
        try:
           
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to read image"
            
          
            faces = self.face_app.get(image)
            
            if not faces:
                return None, "No face detected in image"
            
            if len(faces) > 1:
                return None, "Multiple faces detected, please provide an image with a single face"
            
            
            return faces[0], "Success"
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, f"Error processing image: {str(e)}"

    def detect_face_covering(self, face, image):
        """Detect if a face is covered with mask, sunglasses, etc."""
        try:
            # Get face bounding box
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            # Get facial landmarks
            if not hasattr(face, 'kps') or face.kps.shape[0] < 5:
                return True, "Cannot detect facial landmarks clearly"
            
            landmarks = face.kps
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Calculate regions of interest
            eye_region_height = int((y2 - y1) * 0.2)
            mouth_region_height = int((y2 - y1) * 0.25)
            nose_region_height = int((y2 - y1) * 0.15)
            
            # Eye region detection
            eye_y_center = (left_eye[1] + right_eye[1]) / 2
            eye_region_y1 = max(0, int(eye_y_center - eye_region_height/2))
            eye_region_y2 = min(y2-y1, int(eye_y_center + eye_region_height/2))
            eye_region = face_region[eye_region_y1:eye_region_y2, :]
            
            # Nose region detection
            nose_y = nose[1] - y1
            nose_region_y1 = max(0, int(nose_y - nose_region_height/2))
            nose_region_y2 = min(y2-y1, int(nose_y + nose_region_height/2))
            nose_region = face_region[nose_region_y1:nose_region_y2, :]
            
            # Mouth region detection
            mouth_y_center = ((left_mouth[1] + right_mouth[1]) / 2) - y1
            mouth_region_y1 = max(0, int(mouth_y_center - mouth_region_height/2))
            mouth_region_y2 = min(y2-y1, int(mouth_y_center + mouth_region_height/2))
            mouth_region = face_region[mouth_region_y1:mouth_region_y2, :]
            
            # Convert regions to grayscale for analysis
            if len(face_region.shape) == 3:
                gray_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                gray_nose_region = cv2.cvtColor(nose_region, cv2.COLOR_BGR2GRAY)
                gray_mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye_region = eye_region
                gray_nose_region = nose_region
                gray_mouth_region = mouth_region
            
            # Calculate edge density for each region
            eye_edges = cv2.Canny(gray_eye_region, 50, 150)
            nose_edges = cv2.Canny(gray_nose_region, 50, 150)
            mouth_edges = cv2.Canny(gray_mouth_region, 50, 150)
            
            eye_edge_density = np.sum(eye_edges > 0) / eye_edges.size if eye_edges.size > 0 else 0
            nose_edge_density = np.sum(nose_edges > 0) / nose_edges.size if nose_edges.size > 0 else 0
            mouth_edge_density = np.sum(mouth_edges > 0) / mouth_edges.size if mouth_edges.size > 0 else 0
            
            # Calculate texture variance for each region
            eye_variance = np.var(gray_eye_region) if gray_eye_region.size > 0 else 0
            nose_variance = np.var(gray_nose_region) if gray_nose_region.size > 0 else 0
            mouth_variance = np.var(gray_mouth_region) if gray_mouth_region.size > 0 else 0
            
            # Calculate skin tone ratio for each region
            if len(face_region.shape) == 3:
                hsv_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
                hsv_nose_region = cv2.cvtColor(nose_region, cv2.COLOR_BGR2HSV)
                hsv_mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
                
                # Extended skin tone range
                lower_skin = np.array([0, 15, 60], dtype=np.uint8)
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                
                eye_skin_mask = cv2.inRange(hsv_eye_region, lower_skin, upper_skin)
                nose_skin_mask = cv2.inRange(hsv_nose_region, lower_skin, upper_skin)
                mouth_skin_mask = cv2.inRange(hsv_mouth_region, lower_skin, upper_skin)
                
                eye_skin_ratio = np.sum(eye_skin_mask > 0) / eye_skin_mask.size if eye_skin_mask.size > 0 else 0
                nose_skin_ratio = np.sum(nose_skin_mask > 0) / nose_skin_mask.size if nose_skin_mask.size > 0 else 0
                mouth_skin_ratio = np.sum(mouth_skin_mask > 0) / mouth_skin_mask.size if mouth_skin_mask.size > 0 else 0
            else:
                eye_skin_ratio = 0
                nose_skin_ratio = 0
                mouth_skin_ratio = 0
            
            # Check for covered eyes (sunglasses detection)
            if eye_edge_density < 0.03 and eye_variance < 100 and eye_skin_ratio < 0.3:
                return True, "Eyes appear to be covered, possibly wearing sunglasses"
                
            # Check for covered mouth and nose (mask detection)
            if mouth_edge_density < 0.04 and mouth_variance < 100 and mouth_skin_ratio < 0.3:
                return True, "Mouth appears to be covered, possibly wearing a mask"
                
            if nose_edge_density < 0.04 and nose_variance < 100 and nose_skin_ratio < 0.3:
                return True, "Nose appears to be covered, possibly wearing a mask"
            
            # Additional check for unnatural color patterns that might indicate face covering
            if len(face_region.shape) == 3:
                # Calculate color histograms
                color_regions = [eye_region, nose_region, mouth_region]
                color_histograms = []
                
                for region in color_regions:
                    if region.size == 0:
                        continue
                    hist_b = cv2.calcHist([region], [0], None, [32], [0, 256])
                    hist_g = cv2.calcHist([region], [1], None, [32], [0, 256])
                    hist_r = cv2.calcHist([region], [2], None, [32], [0, 256])
                    
                    # Normalize histograms
                    if np.sum(hist_b) > 0:
                        hist_b = hist_b / np.sum(hist_b)
                    if np.sum(hist_g) > 0:
                        hist_g = hist_g / np.sum(hist_g)
                    if np.sum(hist_r) > 0:
                        hist_r = hist_r / np.sum(hist_r)
                    
                    color_histograms.append((hist_b, hist_g, hist_r))
                
                # Check for unusual color distributions
                for hist_b, hist_g, hist_r in color_histograms:
                    # Look for sharp peaks in color distribution that might indicate synthetic materials
                    if np.max(hist_b) > 0.3 or np.max(hist_g) > 0.3 or np.max(hist_r) > 0.3:
                        # Check if the peak is isolated (characteristic of uniform colored masks)
                        sorted_b = np.sort(hist_b.flatten())
                        sorted_g = np.sort(hist_g.flatten())
                        sorted_r = np.sort(hist_r.flatten())
                        
                        if (sorted_b[-1] > 2.5 * sorted_b[-2] or 
                            sorted_g[-1] > 2.5 * sorted_g[-2] or 
                            sorted_r[-1] > 2.5 * sorted_r[-2]):
                            return True, "Unusual color pattern detected, possibly face covering"
            
            # Face appears uncovered
            return False, "No face covering detected"
            
        except Exception as e:
            logger.error(f"Error in face covering detection: {e}")
            # If there's an error, we'll be cautious and assume there might be an issue
            return True, f"Error analyzing face covering: {str(e)}"

    def check_face_quality(self, face, image):
        """Check if the full face is visible and not occluded - with more lenient quality thresholds"""
        try:
            # Get face bounding box
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Basic check: ensure face is completely in frame
            img_h, img_w = image.shape[:2]
            if x1 < 0 or y1 < 0 or x2 >= img_w or y2 >= img_h:
                return False, "Face is partially out of frame"
            
            # Reduced minimum size check for low-quality images (reduced from 60 to 40)
            face_width = x2 - x1
            face_height = y2 - y1
            if face_width < 40 or face_height < 40:  # More lenient size requirement
                return False, "Face is too small in the image, please provide a clearer photo"
            
            # Reduced confidence threshold for face detection (reduced from 0.7 to 0.5)
            if hasattr(face, 'det_score') and face.det_score < 0.5:
                return False, "Face cannot be clearly detected, please try another photo"
            
            # Extract face region for additional analysis
            face_region = image[y1:y2, x1:x2]
            
            # First check specifically for face covering
            is_covered, covering_message = self.detect_face_covering(face, image)
            if is_covered:
                return False, covering_message
            
            # Check if key facial landmarks are present and within image
            if hasattr(face, 'kps'):
                landmarks = face.kps
                # Check if any landmarks are outside the image
                for point in landmarks:
                    x, y = point
                    if x < 0 or y < 0 or x >= img_w or y >= img_h:
                        return False, "Part of the face appears to be cut off"
                
                if len(landmarks) >= 5:
                    left_eye = landmarks[0]
                    right_eye = landmarks[1]
                    nose = landmarks[2]
                    left_mouth = landmarks[3]
                    right_mouth = landmarks[4]
                    
                    # Check if both eyes and mouth are detected
                    if not all([left_eye.any(), right_eye.any(), nose.any(), left_mouth.any(), right_mouth.any()]):
                        return False, "Some parts of the face are not visible"
                    
                    # More lenient head rotation check (increased from 25 to 35 degrees)
                    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                    if abs(eye_angle) > 35:
                        return False, "Face is too tilted, please provide a more straight-facing photo"
                    
                    # More lenient landmark visibility check
                    def check_landmark_visibility(point, radius=15):
                        x, y = point
                        x, y = int(x), int(y)
                        
                        # Convert to image-relative coordinates
                        x_rel = x - x1
                        y_rel = y - y1
                        
                        # Ensure the point is within bounds
                        if (x_rel - radius < 0 or y_rel - radius < 0 or 
                            x_rel + radius >= face_width or y_rel + radius >= face_height):
                            return False
                        
                        # Extract region around landmark
                        landmark_region = face_region[max(0, y_rel-radius):min(face_height, y_rel+radius), 
                                                    max(0, x_rel-radius):min(face_width, x_rel+radius)]
                        
                        # More lenient variance check (reduced from 15 to 10)
                        if landmark_region.size > 0:
                            std_dev = np.std(landmark_region)
                            if std_dev < 10:  # Lower threshold for variance
                                return False
                        return True
                    
                    # Check visibility for key landmarks
                    key_landmarks = [left_eye, right_eye, nose]  # Only check critical landmarks
                    landmarks_visible = [check_landmark_visibility(lm) for lm in key_landmarks]
                    
                    if not all(landmarks_visible):
                        return False, "Critical facial features appear to be covered or occluded"
                    
                    # More lenient face proportion check
                    eye_distance = np.linalg.norm(right_eye - left_eye)
                    nose_to_mouth = np.linalg.norm(nose - ((left_mouth + right_mouth) / 2))
                    
                    # Wider acceptable range for face proportions
                    if nose_to_mouth < 0.2 * eye_distance or nose_to_mouth > 1.0 * eye_distance:
                        return False, "Face proportions appear abnormal, possibly due to occlusion"
            
            # Occlusion detection - still strict because we want to ensure face isn't covered
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
                
            # More lenient edge detection for low quality images
            edges = cv2.Canny(gray_face, 40, 120)  # Adjusted thresholds
            edge_ratio = np.sum(edges > 0) / (face_width * face_height)
            
            # More lenient edge ratio threshold (increased from 0.15 to 0.25)
            if edge_ratio > 0.25:
                return False, "Something appears to be blocking the face"
            
            # More lenient skin tone check
            if len(face_region.shape) == 3:
                hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                
                # Expanded skin tone range to account for different lighting and ethnicities
                lower_skin = np.array([0, 15, 60], dtype=np.uint8)  # More lenient parameters
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)  # Expanded hue range
                
                skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
                
                # Lower threshold for skin detection (reduced from 0.4 to 0.3)
                skin_ratio = np.sum(skin_mask > 0) / (face_width * face_height)
                
                if skin_ratio < 0.3:
                    return False, "Face appears to be partially covered"
            
            # If all checks pass, face is acceptable
            return True, "Face check passed"
            
        except Exception as e:
            logger.error(f"Error checking face quality: {e}")
            return False, f"Error checking face quality: {str(e)}"
    
    def validate_face_image(self, image_path):
        """Validate if the image contains a clear face"""
        face, message = self.process_image(image_path)
        
        if face is None:
            return False, message
        
        # Check face quality
        image = cv2.imread(image_path)
        is_quality_face, quality_message = self.check_face_quality(face, image)
        if not is_quality_face:
            return False, quality_message
        
        # Check for duplicate face
        embedding = face.normed_embedding
        closest_match, distance = self.find_closest_match(embedding, threshold=0.4)
        
        if closest_match:
            return False, "This face already exists in the database"
        
        return True, "Face image is valid and unique"
    
    def find_closest_match(self, embedding, threshold=0.5):
        """Find the closest face match in the database"""
        try:
           
            all_faces = list(self.collection.find())
            
            if not all_faces:
                return None, float('inf')
            
            closest_match = None
            min_distance = float('inf')
            
            for face_doc in all_faces:
                if 'embedding' in face_doc:
                     
                    stored_embedding = pickle.loads(face_doc['embedding'])
                    
                  
                    distance = 1 - np.dot(embedding, stored_embedding)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = face_doc
             
            if min_distance <= threshold:
                return closest_match, min_distance
            else:
                return None, min_distance
                
        except Exception as e:
            logger.error(f"Error finding closest match: {e}")
            return None, float('inf')
    
    def store_face(self, image_path):
        """Store a face embedding in the database"""
        face, message = self.process_image(image_path)
        
        if face is None:
            return False, message
            
        # Check face quality before storing
        image = cv2.imread(image_path)
        is_quality_face, quality_message = self.check_face_quality(face, image)
        if not is_quality_face:
            return False, quality_message
       
        embedding = face.normed_embedding
        
        try:
            
            existing_face, distance = self.find_closest_match(embedding, threshold=0.4)
            if existing_face:
                return False, "This face appears to be already registered"
         
            embedding_binary = Binary(pickle.dumps(embedding))
          
            doc = {
                'user_id': str(uuid.uuid4()),
                'embedding': embedding_binary,
                'timestamp': time.time()
            }
          
            result = self.collection.insert_one(doc)
            logger.info(f"Successfully stored face with ID: {result.inserted_id}")
            
            return True, f"Face stored successfully with user_id: {doc['user_id']}"
            
        except Exception as e:
            logger.error(f"Error storing face: {e}")
            return False, f"Error storing face: {str(e)}"
    
    def verify_face(self, image_path, threshold=0.5):
        """Verify a face against the database"""
        face, message = self.process_image(image_path)
        
        if face is None:
            return False, message
            
        # For verification, we still want basic quality checks but can be less strict
        image = cv2.imread(image_path)
        is_quality_face, quality_message = self.check_face_quality(face, image)
        if not is_quality_face:
            return False, quality_message
  
        embedding = face.normed_embedding
 
        closest_match, distance = self.find_closest_match(embedding, threshold)
        
        if closest_match:
         
            user_id = closest_match.get('user_id', '')
            confidence = float(1 - distance)
            return True, f"Face verified successfully with confidence: {confidence:.2f}", user_id
        else:
            return False, "No matching face found", None

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

 
MONGODB_URI = "mongodb+srv://projectDB:PEyHwQ2fF7e5saEf@cluster0.43hxo.mongodb.net/"
DB_NAME = "ta7t-bety"
COLLECTION_NAME = "face_id_images"

face_api = FaceRecognitionAPI(MONGODB_URI, DB_NAME, COLLECTION_NAME)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({'status': 'success', 'message': 'Face Recognition API is running'})

@app.route('/signup', methods=['POST'])
def signup():
    """Endpoint to store a face in the database for signup"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
       
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
        file.save(file_path)
        
     
        is_valid, message = face_api.validate_face_image(file_path)
        
        if is_valid:
             
            success, store_message = face_api.store_face(file_path)
    
            try:
                os.remove(file_path)
            except:
                pass
            
            if success:
                return jsonify({
                    'status': 'success', 
                    'message': store_message
                })
            else:
                return jsonify({
                    'status': 'error', 
                    'message': store_message
                }), 400
        else:
         
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify({
                'status': 'error', 
                'message': message
            }), 400
    
    return jsonify({'status': 'error', 'message': 'Invalid file format. Please use JPG, JPEG or PNG'}), 400

@app.route('/verify', methods=['POST'])
def verify():
    """Endpoint to verify a face against the database"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
  
    threshold = request.form.get('threshold', 0.5)
    try:
        threshold = float(threshold)
    except:
        threshold = 0.5
    
    if file and allowed_file(file.filename):
   
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
        file.save(file_path)
        
       
        verified, message, user_id = face_api.verify_face(file_path, threshold)
        
       
        try:
            os.remove(file_path)
        except:
            pass
        
        if verified:
            return jsonify({
                'status': 'success', 
                'message': message,
                'verified': True,
                'user_id': user_id
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': message,
                'verified': False
            }), 401
    
    return jsonify({'status': 'error', 'message': 'Invalid file format. Please use JPG, JPEG or PNG'}), 400

if __name__ == '__main__':
  
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', default=7000, type=int, help='Port to run the server on')
    parser.add_argument('--mongodb-uri', 
                        default="mongodb+srv://projectDB:PEyHwQ2fF7e5saEf@cluster0.43hxo.mongodb.net/",
                        help='MongoDB connection URI')
    parser.add_argument('--db-name', default="ta7t-bety", help='Database name')
    parser.add_argument('--collection', default="face_id_images", help='Collection name')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    face_api = FaceRecognitionAPI(args.mongodb_uri, args.db_name, args.collection)
    
   
    app.run(host=args.host, port=args.port, debug=args.debug)