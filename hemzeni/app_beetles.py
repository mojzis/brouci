import streamlit as st
from hemzeni.controller import BeetleController


def main() -> None:
    st.set_page_config(layout="wide")

    # UI components
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([], use_container_width=True)

    # Initialize controller in session state
    if "controller" not in st.session_state:
        st.session_state.controller = BeetleController()
        st.session_state.controller.initialize_beetles()

    controller = st.session_state.controller

    # Main loop
    while run:
        result = controller.process_frame()

        if result is None:
            continue

        frame, human_positions, human_bboxes = result

        # Display frame
        FRAME_WINDOW.image(frame)

        # Update status
        # message, status_type = controller.get_status_message(human_positions)
        # if status_type == "success":
        #     st.sidebar.success(message)
        # else:
        #     st.sidebar.info(message)

    else:
        # Stopped state
        st.write("Stopped")

        # if hasattr(st.session_state, "pred_list") and st.session_state.pred_list:
        #     st.write("Last prediction:", st.session_state.pred_list[-1])

        #     if hasattr(st.session_state, "hand"):
        #         st.write("Last hand position:", st.session_state.hand)

        if hasattr(st.session_state, "img"):
            FRAME_WINDOW.image(st.session_state.img)


if __name__ == "__main__":
    main()
