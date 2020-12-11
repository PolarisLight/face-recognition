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
        if faces:
            for face in faces:
                face_distances = database.getFaceDistance(face)
                best_match_index = np.argmin(face_distances)
                # if face_distances[best_match_index] >= threshold:
                print(face_distances[best_match_index])
                names.append(database.known_face_names[best_match_index])
        frame = cv2.resize(frameOpencvDnn, (0, 0),
                           fx=resize_rate,
                           fy=resize_rate)
        if names:
            for (x1, y1, x2, y2), name in zip(bboxes, names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                x1 *= resize_rate
                y1 *= resize_rate
                x2 *= resize_rate
                y2 *= resize_rate

                # Draw a box around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (x1, y2), (x2, y2 - 35), (0, 0, 255), cv2.FILLED)

                frame = DrawChinese(frame, name, (x1 + 6, y2 - 20), fontSize=20, fontColor=(255, 255, 255))
        time = 1.0 / ((cv2.getTickCount() - t) / cv2.getTickFrequency())
        text = "Frame rate: %.2f fps" % time
        cv2.putText(frame, text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit Esc on the keyboard to quit!
        faces.clear()
        names.clear()
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()
