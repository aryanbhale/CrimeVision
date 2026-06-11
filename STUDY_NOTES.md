# 🎓 CRIMINAL RECORD MANAGEMENT SYSTEM
## Complete Study Prep Kit — Final Year Project
---

# TABLE OF CONTENTS
1. Project Overview
2. Tech Stack Explained
3. System Architecture (How Everything Connects)
4. Face Recognition — Deep Dive
5. KNN Algorithm — How Matching Works
6. dlib — The Brain Behind Face Detection
7. Database Structure
8. API Endpoints — Complete Reference
9. Complete Flow: Add Record → Train → Match
10. Key Code Walkthrough (Showable in Presentation)
11. Common Questions & Model Answers
12. Quick Revision Cheat Sheet

---

# 1. PROJECT OVERVIEW

## What is this project?
An AI-powered Criminal Record Management System that allows police to:
- Store criminal records (name, photo, cases)
- Upload a suspect's photo
- Automatically identify if that person exists in the criminal database
- Manage citizen requests (complaints, NOCs, appointments)

## The Core Problem It Solves
Manually identifying criminals from photos is slow and error-prone.
This system automates it using Face Recognition + Machine Learning.

## Real-World Use Case
Imagine police have CCTV footage of a robbery suspect.
They take a screenshot and upload it to the system.
Within seconds, the system returns the criminal's name and case history.

---

# 2. TECH STACK EXPLAINED

| Technology       | Version  | Role                                      |
|------------------|----------|-------------------------------------------|
| Python           | 3.10     | Main programming language                 |
| Django           | 3.0.6    | Web framework (handles URLs, views, DB)   |
| Django REST Framework | 3.11.0 | Builds the API endpoints               |
| dlib             | 19.22.99 | Face detection + 128-point face encoding  |
| OpenCV (cv2)     | 4.8.1    | Image resizing and processing             |
| scikit-learn     | 1.3.2    | KNN classifier for face matching          |
| numpy            | 1.26.4   | Numerical arrays (face vectors)           |
| Pillow (PIL)     | 9.5.0    | Opening and converting images             |
| SQLite           | Built-in | Database to store criminal records        |
| jQuery / AJAX    | 3.1.1    | Frontend: sends images without page reload|
| Bootstrap 4      | CDN      | UI styling                                |

---

# 3. SYSTEM ARCHITECTURE

```
[Police Officer]
      |
      | uploads image via browser
      v
[Django Web Server: /police/match/]
      |
      | AJAX POST request with image
      v
[API: /police/api/match/]
      |
      | 1. Opens image with PIL
      | 2. Converts to base64
      | 3. Calls match() function
      v
[find_matches.py → match()]
      |
      | 1. Decodes base64 → numpy array
      | 2. Extracts 128-point face encoding via dlib
      | 3. Loads classifier.pkl (trained KNN model)
      | 4. Compares face encoding with all known faces
      v
[Result]
      |
      |-- Match found → returns name, image, cases
      |-- No match   → "No Records Found"
```

## Two-Part System:
- **Citizen Portal**: Complaints, NOC requests, appointments
- **Police Portal**: Criminal records, face recognition matching

---

# 4. FACE RECOGNITION — DEEP DIVE

## How does the computer "see" a face?

### Step 1: Face Detection
dlib uses a pre-trained HOG (Histogram of Oriented Gradients) model
to find where faces are in an image.

  - HOG looks at which direction pixels change (edges)
  - A face has consistent patterns: eyes, nose, mouth
  - Output: A bounding box (rectangle) around each detected face

### Step 2: Facial Landmark Detection
dlib's shape_predictor_68_face_landmarks.dat finds 68 key points on the face:
  - 6 points around each eye
  - 9 points on the nose
  - 20 points around the mouth
  - 17 points along the jawline
  - etc.

These landmarks help align the face so angle/tilt doesn't matter.

### Step 3: Face Encoding (The "Fingerprint")
dlib's ResNet model (dlib_face_recognition_resnet_model_v1.dat) converts
the aligned face into a list of 128 numbers.

  EXAMPLE:
  Face of Person A = [0.142, -0.053, 0.218, 0.091, ... ] (128 numbers)
  Face of Person B = [0.831, -0.210, 0.514, 0.333, ... ] (128 numbers)

These 128 numbers are called a "face encoding" or "face embedding".
They capture the unique geometry of a face — distance between eyes,
nose shape, jawline width, etc.

  KEY INSIGHT: The SAME person in DIFFERENT photos
  will produce SIMILAR (but not identical) 128-number vectors.
  DIFFERENT people produce VERY DIFFERENT vectors.

### Step 4: Comparison
To check if two faces match, calculate the Euclidean distance
between their 128-number vectors:

  Distance = sqrt( (a1-b1)² + (a2-b2)² + ... + (a128-b128)² )

  Distance < 0.5  → Same person  ✅
  Distance > 0.5  → Different person ❌

This threshold (0.5) is the magic number used in this project.

---

# 5. KNN ALGORITHM — HOW MATCHING WORKS

## What is KNN?
K-Nearest Neighbors. A simple but powerful classification algorithm.

## The Concept (Simple Analogy)
Imagine you're in a room full of people holding colored flags.
You walk in — which color flag should YOU hold?
KNN says: "Look at your K nearest neighbors and take their most common flag."

## How This Project Uses KNN

### During Training (/api/train/):
1. Load all criminal records from DB
2. Decode each stored key_points (face encoding) back to 128 numbers
3. Label them with their unique criminal key (e.g., "CR_1", "CR_2")
4. Train KNN with n_neighbors = total records

  ```python
  classifier = KNeighborsClassifier(
      n_neighbors=len(labels),   # considers all criminals
      algorithm='ball_tree',      # efficient spatial search
      weights='distance'          # closer = more weight
  )
  classifier.fit(key_pts, encoded_labels)
  ```

5. Save the trained model as classifier.pkl (pickle file)

### During Matching (/api/match/):
1. Extract 128-number encoding from uploaded image
2. Ask KNN: "Who is the nearest neighbor to this encoding?"
3. KNN returns the closest criminal record
4. Check: is the distance < 0.5?
   - YES → Match found, return criminal's details
   - NO  → Too far, likely not in database

  ```python
  closest_distances = clf.kneighbors(key_pts)
  is_recognized = [closest_distances[0][0][0] <= 0.5]
  ```

## Why ball_tree?
Ball tree organizes data into nested hyperspheres (balls) for fast nearest-neighbor search.
Much faster than brute-force comparison when database grows.

---

# 6. DLIB — THE BRAIN BEHIND FACE RECOGNITION

## What is dlib?
A C++ library with Python bindings for machine learning and computer vision.
Created by Davis King. Free and open source.

## Two Pre-Trained Models Used:

### Model 1: shape_predictor_68_face_landmarks.dat (99 MB)
- Trained on thousands of human face images
- Detects 68 specific landmark points on any face
- Used to align and normalize the face before encoding

### Model 2: dlib_face_recognition_resnet_model_v1.dat (22 MB)
- A deep ResNet (Residual Neural Network) with 29 convolutional layers
- Trained on 3 million face images
- Converts any face into a 128-dimensional vector
- Achieves 99.38% accuracy on standard benchmarks

## How This Project Loads Them:

  ```python
  # In face_recognition.py
  face_detector = dlib.get_frontal_face_detector()
  pose_predictor = dlib.shape_predictor('police/files/shape_predictor_68_face_landmarks.dat')
  face_encoder = dlib.face_recognition_model_v1('police/files/dlib_face_recognition_resnet_model_v1.dat')
  ```

These load ONCE when the Django server starts (module-level).

## Why dlib Instead of OpenCV Face Recognition?
dlib's ResNet model is significantly more accurate for face identification.
OpenCV's Haar Cascades are good for detection but not for matching identity.

---

# 7. DATABASE STRUCTURE

## Model: CrimeRecords

  ```
  Table: police_crimerecords
  ┌─────────────┬──────────────┬───────────────────────────────────────┐
  │ Field       │ Type         │ Description                           │
  ├─────────────┼──────────────┼───────────────────────────────────────┤
  │ key         │ CharField PK │ Unique criminal ID (e.g., "CR_001")   │
  │ name        │ CharField    │ Criminal's full name                  │
  │ against     │ CharField    │ Number of cases filed against them    │
  │ gender      │ CharField    │ Male / Female                         │
  │ img         │ Base64Field  │ Photo stored as base64 string         │
  │ key_points  │ CharField    │ 128-d face encoding as custom string  │
  └─────────────┴──────────────┴───────────────────────────────────────┘
  ```

## How key_points is Stored (Custom Encoding)

The 128 floats can't be stored as a Python list in SQLite.
The project encodes them as a custom string:

  EXAMPLE (simplified):
  Face encoding: [0.142, -0.053, 0.218]

  Custom encoding steps:
  1. Replace '.'  with '$'    → 0$142
  2. Replace '-'  with '1'    → 10$053 (negative number)
  3. Join with '@'             → @0$142@10$053@0$218

  Stored in DB as: "@0$142@10$053@0$218"

  To decode back:
  1. Split by '@'
  2. Replace '$' → '.'
  3. If starts with '1' → add '-' and remove the '1'
  4. Convert to float

  ```python
  # face_encoding.py - encode()
  for value in key_points[0]:
      svalue = str(value)
      if value < 0:
          svalue = svalue.replace('-', '1')
      svalue = svalue.replace('.', '$')
      encoded_string += '@' + svalue

  # face_encoding.py - decode()
  for t in text:
      t = t.replace('$', '.')
      if t[0:1] == '1':
          t = '-' + t[1:]
      encoded.append(float(t))
  ```

---

# 8. API ENDPOINTS — COMPLETE REFERENCE

Base URL: http://localhost:8000/police/

| Method | Endpoint              | Auth | Description                        |
|--------|-----------------------|------|------------------------------------|
| GET    | api/                  | Yes  | API overview / list of endpoints   |
| GET    | api/description/      | Yes  | What this API does                 |
| GET    | api/records/          | Yes  | Fetch all criminal records         |
| POST   | api/add-record/       | Yes  | Add a new criminal record          |
| DELETE | api/delete-record/pk/ | Yes  | Delete a criminal record           |
| GET    | api/train/            | Yes  | Train KNN model from DB records    |
| POST   | api/match/            | Yes  | Match uploaded image with DB       |

### POST api/add-record/ — Required Fields:
```
key      : unique ID string (e.g., "CR_005")
name     : full name string
against  : number of cases (number)
gender   : "Male" or "Female"
file     : image file (multipart/form-data)
```

### POST api/match/ — Required Fields:
```
file     : image file (multipart/form-data)
csrfmiddlewaretoken : Django CSRF token
```

### Response from api/match/ on success:
```json
{
  "CR_1": {
    "name": "John Doe",
    "against": "3",
    "img": "b'/9j/4AAQSkZJRgAB...'"
  }
}
```

---

# 9. COMPLETE FLOW: ADD RECORD → TRAIN → MATCH

```
STEP 1 — ADD CRIMINAL RECORD
==============================
Police uploads: photo + name + ID + cases

System does:
  a) Opens image with PIL
  b) Converts to base64 bytes
  c) Extracts 128-point face encoding via dlib
  d) Encodes to custom string format
  e) Saves to SQLite DB (img + key_points)

API: POST /police/api/add-record/
Result: Record saved in CrimeRecords table


STEP 2 — TRAIN THE MODEL
==========================
Must be done after adding/updating records.

System does:
  a) Fetches all records from DB
  b) Decodes each key_points string → list of 128 floats
  c) Labels each with the criminal's 'key'
  d) Trains KNeighborsClassifier
  e) Saves to classifier.pkl using pickle

API: GET /police/api/train/
CLI: python manage.py retrain
Result: classifier.pkl saved in project root


STEP 3 — MATCH A SUSPECT IMAGE
================================
Police uploads a photo of a suspect.

System does:
  a) Opens image with PIL → RGB numpy array
  b) dlib detects face in the image
  c) Extracts 128-point face encoding
  d) Loads classifier.pkl
  e) KNN finds nearest neighbor in trained data
  f) Checks: distance <= 0.5?
     → YES: Returns criminal name + photo + cases
     → NO:  Returns "No Records Found"

API: POST /police/api/match/
Result: Matched criminal's details (JSON)
```

---

# 10. KEY CODE WALKTHROUGH (Show These in Presentation)

## A) Face Encoding Extraction (face_recognition.py)

```python
def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Takes an image (numpy array) and returns a list of 128-dimensional
    face encodings — one per face detected in the image.
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
    return [
        np.array(face_encoder.compute_face_descriptor(face_image, landmark, num_jitters))
        for landmark in raw_landmarks
    ]
```
EXPLAIN: compute_face_descriptor() is the dlib ResNet call.
It takes the aligned face and outputs the 128-number vector.

---

## B) KNN Training (train.py)

```python
def start_train():
    labels, key_pts = fetch_data()          # load from DB
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels) # "CR_1" → 0, "CR_2" → 1, etc.
    
    classifier = KNeighborsClassifier(
        n_neighbors=len(labels),
        algorithm='ball_tree',
        weights='distance'
    )
    classifier.fit(key_pts, encoded_labels)
    
    with open('classifier.pkl', 'wb') as f:
        pickle.dump((le, classifier), f)     # save both encoder + model
```
EXPLAIN: LabelEncoder converts text labels to numbers.
pickle serializes the trained model to disk.

---

## C) Matching Logic (find_matches.py)

```python
def match(base64_image):
    with open('classifier.pkl', 'rb') as f:
        (le, clf) = pickle.load(f)           # load trained model
    
    image = decode_base64(str(base64_image)) # base64 → numpy array
    key_pts = face_encodings(image)          # extract 128-d encoding
    
    if not key_pts:
        return []                             # no face detected
    
    closest_distances = clf.kneighbors(key_pts)
    distance = closest_distances[0][0][0]    # distance to nearest neighbor
    
    if distance <= 0.5:                      # threshold check
        prediction = le.inverse_transform(   # number → "CR_1"
            clf.predict(key_pts)
        )
        return [[[prediction]]]
    return []                                # too far = no match
```
EXPLAIN: kneighbors() returns distances + indices of nearest neighbors.
0.5 is the Euclidean distance threshold — tuned by dlib researchers.

---

## D) AJAX Call in Frontend (match.html)

```javascript
$('#match').click(function() {
    var data = new FormData();
    data.append('file', $('#image')[0].files[0]);          // the image
    data.append('csrfmiddlewaretoken', '{{ csrf_token }}'); // Django security

    $.ajax({
        url: '/police/api/match/',
        type: 'POST',
        data: data,
        processData: false,    // don't convert FormData
        contentType: false,    // let browser set multipart boundary
        success: function(response) {
            // display matched criminal's name and photo
            for (var key in response) {
                console.log("Matched: " + response[key].name);
            }
        }
    });
});
```
EXPLAIN: FormData lets us send binary image files via AJAX.
processData:false and contentType:false are required for file uploads.

---

# 11. COMMON QUESTIONS & MODEL ANSWERS

## Q1: Why use dlib instead of OpenCV for face recognition?
**A:** OpenCV's face recognition (Eigenfaces, LBPH) is older and less accurate.
dlib uses a deep ResNet neural network trained on 3 million faces achieving
99.38% accuracy on LFW benchmark. For a criminal identification system,
accuracy is critical — dlib was the right choice.

## Q2: What is a 128-dimensional face encoding?
**A:** It's a list of 128 floating-point numbers that mathematically represent
the unique geometric features of a face — like the distance between your eyes,
the width of your nose, the shape of your jaw. Two photos of the same person
will produce similar vectors; different people produce very different vectors.

## Q3: Why do you need to train after adding records?
**A:** The KNN classifier needs to be rebuilt every time the database changes.
It stores the face encodings in memory in a spatial data structure (ball tree)
for fast searching. classifier.pkl is the serialized trained model that the
match function loads during recognition.

## Q4: What is the 0.5 threshold?
**A:** It's the maximum Euclidean distance allowed between two face encodings
to consider them the same person. Below 0.5 = match. Above 0.5 = no match.
This value was established through extensive testing by dlib's researchers.
If we set it lower (e.g., 0.3), fewer false positives but more false negatives.

## Q5: What happens if two faces are found in one image?
**A:** face_encodings() returns a list — one encoding per face detected.
The current implementation matches only the first face (key_pts[0]).
For a production system, you'd loop through all faces.

## Q6: How is the image stored in the database?
**A:** The image is converted to base64 bytes using Python's base64 module,
then stored as a string in the Base64Field. When retrieved, it's sent back
to the frontend as a base64 string which the browser can display as:
  <img src="data:image/png;base64,/9j/4AAQ...">

## Q7: What is pickle / classifier.pkl?
**A:** Pickle is Python's built-in serialization module. It converts any Python
object (like our trained KNN model) into a binary file format that can be saved
to disk and loaded back later. classifier.pkl contains both:
  1. The LabelEncoder (maps numbers back to criminal IDs)
  2. The KNeighborsClassifier (the trained matching model)

## Q8: Why base64 for image storage instead of a file path?
**A:** The original developers chose base64 to keep everything self-contained
in the database without managing a file system. The trade-off is larger DB size.
A production system would typically use a file path or cloud storage URL.

## Q9: What is CSRF and why does the AJAX call need it?
**A:** CSRF (Cross-Site Request Forgery) protection is a Django security feature.
Every POST request must include a secret token to prove it came from the real
website, not a malicious third party. Django generates this token and embeds it
in forms using {% csrf_token %}.

## Q10: How does Django REST Framework fit in?
**A:** DRF is a toolkit that makes building web APIs easy in Django. It provides:
  - @api_view decorator to specify allowed HTTP methods
  - Response() class that automatically serializes Python dicts to JSON
  - Status codes (HTTP_200_OK, HTTP_400_BAD_REQUEST, etc.)

---

# 12. QUICK REVISION CHEAT SHEET

```
PROJECT TYPE   : Django Web App + Face Recognition AI + REST API
LANGUAGE       : Python 3.10
DATABASE       : SQLite (db.sqlite3)
AI ALGORITHM   : KNN (K-Nearest Neighbors) with dlib face encodings
FACE ENCODING  : 128-dimensional vector per face
SIMILARITY     : Euclidean distance (threshold: 0.5)
MODEL FILES    : shape_predictor_68_face_landmarks.dat
                 dlib_face_recognition_resnet_model_v1.dat
TRAINED MODEL  : classifier.pkl (pickle file, ball tree KNN)
FRONTEND       : HTML + Bootstrap 4 + jQuery AJAX

KEY FILES:
  police/face_recognition.py → dlib integration, extract encodings
  police/face_encoding.py    → encode/decode 128-d vector ↔ string
  police/find_matches.py     → load classifier, run matching
  police/train.py            → build & save KNN from DB records
  police/api_views.py        → REST API views (add/train/match)
  police/urls.py             → URL routing
  police/templates/police/match.html → the main UI page

CORRECT RUN COMMAND:
  .\env\Scripts\python.exe manage.py runserver

WORKFLOW:
  1. Add record (POST /api/add-record/) → stores encoding in DB
  2. Train model (GET /api/train/ or manage.py retrain) → saves .pkl
  3. Match face (POST /api/match/) → returns criminal record

FACE RECOGNITION PIPELINE:
  Image → PIL (convert RGB) → numpy array → dlib HOG (detect face)
  → dlib landmarks (68 points) → dlib ResNet (128-d encoding)
  → KNN classifier → Euclidean distance → Match or No Match
```

---

*Study Notes prepared for: Criminal Record Management System — Final Year Project*
*Good luck with your presentation! 🎓*
