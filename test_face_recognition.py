import csv
import glob

import face_recognition
import tqdm
from cv2 import *

from utils import *

cap = cv2.VideoCapture(0, CAP_DSHOW)
cap.set(CAP_PROP_EXPOSURE, -6)
known_face_encodings = []
known_face_names = []
resize_rate = 4


# initial function
def init():
    global known_face_encodings, known_face_names
    file_dir = "face_database\\"
    read_file = glob.glob(file_dir + "*")
    lines = 0
    if os.path.exists('face_encode.csv'):
        with open('face_encode.csv') as f:
            f_csv = csv.reader(f)
            for _ in f_csv:
                lines += 1
            f.close()
        with open('face_encode.csv') as f:
            f_csv = csv.reader(f)
            with tqdm.tqdm(total=lines) as pbar:
                for row in f_csv:
                    known_face_names.append(row[0])
                    face_encoding = []
                    for x in row[1:]:
                        face_encoding.append(float(x))
                    known_face_encodings.append(face_encoding)
                    pbar.set_description("now loading %s" % row[0])
                    pbar.update(1)
            f.close()
    with open('face_encode.csv', 'a', newline='') as f:
        f_csv = csv.writer(f)
        with tqdm.tqdm(total=len(read_file)) as pbar:
            for imgname in read_file:
                list = imgname.split("\\")
                filename = list[-1]
                name = filename.split(".")[0]
                if name not in known_face_names:
                    img = face_recognition.load_image_file(imgname)
                    face_encoding = face_recognition.face_encodings(img)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    pbar.set_description("now writing %s" % name)
                    row = [name]
                    for x in face_encoding:
                        row.append(x)
                    f_csv.writerow(row)
                else:
                    pbar.set_description("skipping %s" % name)
                pbar.update(1)
        f.close()


def main():
    init()
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        t = getTickCount()
        # Grab a single frame of video
        ret, frame = cap.read()
        if not ret:
            continue
        face_recognition.face_locations()
        # frame = imlocalbrighten(frame)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / resize_rate, fy=1 / resize_rate)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print(face_distances)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= resize_rate
            right *= resize_rate
            bottom *= resize_rate
            left *= resize_rate

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            frame = DrawChinese(frame, name, (left + 6, bottom - 20), fontSize=20, fontColor=(255, 255, 255))

        time = 1.0 / ((getTickCount() - t) / getTickFrequency())
        text = "Frame rate: %.2f fps" % time
        cv2.putText(frame, text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit Esc on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release handle to the camera
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
