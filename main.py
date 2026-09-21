import cv2
import face_recognition
import os

# Folder containing known face images
KNOWN_FACES_DIR = "knownfaces"

known_face_encodings = []
known_face_names = []

# Load all known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

print("Known faces loaded:", known_face_names)

# Start webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown Person"

        # Check if known faces exist
        if len(known_face_encodings) > 0:

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            best_match_index = face_distances.argmin()

            # Accuracy threshold
            if face_distances[best_match_index] < 0.50:
                name = known_face_names[best_match_index]

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()