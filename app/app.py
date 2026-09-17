import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unlearning.unlearning import ModelUnlearner


st.set_page_config(
    page_title="Battlefield Drone",
    page_icon="🚁",
    layout="wide"
)


st.title("Battlefield Drone Analysis System")

st.write(
    "Software-based UAV detection, tracking, "
    "geolocation and machine unlearning prototype."
)


st.sidebar.title("Modules")


module = st.sidebar.selectbox(
    "Select module",
    [
        "Dashboard",
        "Detection",
        "Tracking",
        "Geolocation",
        "Machine Unlearning"
    ]
)


if module == "Dashboard":

    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Detection", "Ready")

    with col2:
        st.metric("Tracking", "Ready")

    with col3:
        st.metric("Geolocation", "Ready")

    st.info(
        "Select a module from the sidebar."
    )


elif module == "Detection":

    st.header("Detection")

    st.write(
        "Person 1's YOLO detection module "
        "will be connected here."
    )


elif module == "Tracking":

    st.header("Object Tracking")

    st.write(
        "Person 2's DeepSORT tracking module "
        "will be connected here."
    )


elif module == "Geolocation":

    st.header("Geolocation")

    st.write(
        "Person 2's geolocation module "
        "will be connected here."
    )


elif module == "Machine Unlearning":

    st.header("Machine Unlearning")

    st.write(
        "Select data or a class that should be "
        "removed from the model."
    )

    data_to_forget = st.text_input(
        "Enter data/class to forget"
    )

    if st.button("Start Unlearning"):

        if data_to_forget:

            unlearner = ModelUnlearner()

            result = unlearner.forget(
                data_to_forget
            )

            st.success(
                result["message"]
            )

            st.write(
                "Selected for forgetting:",
                result["forgotten"]
            )

        else:

            st.warning(
                "Please enter something to forget."
            )