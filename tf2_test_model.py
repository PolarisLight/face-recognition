from utils import *


def main(resize_rate=4, threshold=0.7):
    model = tf.keras.models.load_model("save_model")

    database = FaceDatabase(encodefunc=model)

    detecter = FaceDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        flag, img = cap.read()
        if not flag:
            break
        t = cv2.getTickCount()
        small_frame = cv2.resize(img, (0, 0), fx=1 / resize_rate, fy=1 / resize_rate)
        # rgb_small_frame = small_frame[:, :, ::-1]
        frameOpencvDnn, bboxes, faces = detecter.detectFace(small_frame)
        names = []
        for face in faces:
            face_distances = database.getFaceDistance(face)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] >= threshold:
                name.append(database.known_face_names[best_match_index])
        frame = cv2.resize(frameOpencvDnn, (0, 0),
                           fx=resize_rate,
                           fy=resize_rate)
        for (top, right, bottom, left), name in zip(bboxes, names):
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
        time = 1.0 / ((cv2.getTickCount() - t) / cv2.getTickFrequency())
        text = "Frame rate: %.2f fps" % time
        cv2.putText(frame, text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit Esc on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break
