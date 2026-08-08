
import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2



SQUARES_X = 7                 
SQARES_Y = 5                 

SQUARE_LENGTH = 0.0388           # (m)
MARKER_LENGTH = 0.0197        # (m)

ARUCO_DICT = cv2.aruco.DICT_6X6_250

SAVE_FOLDER = "calibration_images"

# =====================================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    detector_params
)

# ===========================
# Camera
# ===========================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640,480),
        "format":"RGB888"
    }
)

picam2.configure(config)

picam2.start()

picam2.set_controls({

    "AeEnable": False,
    "ExposureTimeMode": 1,
    "ExposureTime": 6000,
    "Sharpness": 1.5,
    "Contrast": 1.2,
    "AnalogueGainMode": 1,
    "NoiseReductionMode": 0,
    "AnalogueGain": 2,
    "FrameDurationLimits": (15000, 15000),
    "AfMode": 0, "LensPosition": 1.0,
    "AwbEnable": True, 

})

time.sleep(0.5)

print("SPACE : Save image")
print("C     : Calibrate")
print("Q     : Quit")

all_charuco_corners = []
all_charuco_ids = []

image_size = None

saved = 0

prev = time.time()

while True:

    frame = picam2.capture_array()

    image_size = frame.shape[:2]

    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    display = frame.copy()

    if ids is not None:

        cv2.aruco.drawDetectedMarkers(display,corners,ids)

        retval, charucoCorners, charucoIds = \
            cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                board
            )

        if retval > 0:

            cv2.aruco.drawDetectedCornersCharuco(
                display,
                charucoCorners,
                charucoIds,
                (0,0,255)
            )

    fps = 1/(time.time()-prev)
    prev = time.time()

    cv2.putText(display,
                f"FPS : {fps:.1f}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.putText(display,
                f"Saved : {saved}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.imshow("Charuco Calibration",display)

    key = cv2.waitKey(1)&0xff

    if key==ord(' '):

        if ids is None:

            print("No board detected.")

            continue

        retval, charucoCorners, charucoIds = \
            cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                board
            )

        if retval < 10:

            print("Too few corners.")

            continue

        filename = os.path.join(
            SAVE_FOLDER,
            f"img_{saved:03d}.png"
        )

        cv2.imwrite(filename,frame)

        all_charuco_corners.append(charucoCorners)
        all_charuco_ids.append(charucoIds)

        saved += 1

        print("Saved",filename)

    elif key==ord('c'):

        if len(all_charuco_corners)<10:

            print("Need at least 10 images.")

            continue

        print("\nCalibrating...\n")

        ret, cameraMatrix, distCoeffs, rvecs, tvecs = \
            cv2.aruco.calibrateCameraCharuco(

                charucoCorners=all_charuco_corners,

                charucoIds=all_charuco_ids,

                board=board,

                imageSize=image_size[::-1],

                cameraMatrix=None,

                distCoeffs=None

            )

        print("\nDone.\n")

        print("RMS Error :",ret)

        print("\nCamera Matrix\n")

        print(cameraMatrix)

        print("\nDistortion\n")

        print(distCoeffs)

        np.savetxt(
            "cameraMatrix.txt",
            cameraMatrix,
            delimiter=","
        )

        np.savetxt(
            "cameraDistortion.txt",
            distCoeffs,
            delimiter=","
        )

        fs=cv2.FileStorage(
            "camera.yaml",
            cv2.FILE_STORAGE_WRITE
        )

        fs.write("camera_matrix",cameraMatrix)

        fs.write("distortion_coefficients",distCoeffs)

        fs.release()

        print("\nSaved.")

    elif key==ord('q'):

        break

picam2.stop()

cv2.destroyAllWindows()

