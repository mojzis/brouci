import cv2
import numpy as np

# import pandas as pd
import streamlit as st
from aipose.models.yolov7.domain import YoloV7Pose
from aipose.plot import plot


def main():
    st.title("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    model = YoloV7Pose()
    if "pred_list" not in st.session_state:
        st.session_state.pred_list = []
        # st.session_state.hand = run
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = model(frame)
        # annotated_image = draw_landmarks_on_image(frame, detection_result)
        pread_array = np.array([value.raw_keypoint for value in keypoints])
        new_frame = plot(
            frame,
            pread_array,
            plot_image=False,
            return_img=True,
        )
        FRAME_WINDOW.image(new_frame)
        st.session_state.pred_list.append(pread_array)
        st.session_state.hand = keypoints[0].get_keypoint(9)
        st.session_state.img = new_frame
    else:
        st.write("Stopped")
        st.write(st.session_state.pred_list[-1])
        for idx in range(st.session_state.pred_list[-1].shape[0]):
            st.write(st.session_state.pred_list[-1][idx])
        st.write(st.session_state.hand)
        FRAME_WINDOW.image(st.session_state.img)
        # st.dataframe(pd.DataFrame(st.session_state.pred_list[-1]))


if __name__ == "__main__":
    main()
