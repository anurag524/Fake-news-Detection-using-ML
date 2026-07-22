import streamlit as st
import joblib
import time

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------
# Load Model
# -------------------------------------
@st.cache_resource
def load_files():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model


vectorizer, model = load_files()

# -------------------------------------
# Session State
# -------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "real_count" not in st.session_state:
    st.session_state.real_count = 0

if "fake_count" not in st.session_state:
    st.session_state.fake_count = 0

# -------------------------------------
# Sidebar
# -------------------------------------
st.sidebar.title("📰 Fake News Detector")

page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "History", "Statistics", "About Us"],
    help="Navigate through different sections."
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Instructions**
    - Paste a news article.
    - Click Analyze News.
    - View results instantly.
    - Check History and Statistics.
    """
)

# -------------------------------------
# HOME PAGE
# -------------------------------------
if page == "Home":

    st.title("📰 Fake News Detector")

    st.write(
        "Analyze and verify news content using advanced Machine Learning and Natural Language Processing techniques to determine its credibility and identify potential misinformation."
    )

    st.divider()

    news_input = st.text_area(
        "Enter News Article",
        height=250,
        placeholder="Paste your news article here...",
        help="Enter a news article or headline for analysis."
    )

    if st.button(
        "Analyze News",
        help="Click to analyze the news article."
    ):

        if news_input.strip():

            with st.spinner("Analyzing News..."):
                time.sleep(1)

                # TF-IDF
                transformed_input = vectorizer.transform(
                    [news_input]
                )

                # Prediction
                prediction = model.predict(
                    transformed_input
                )

                confidence = None

                if hasattr(model, "predict_proba"):
                    confidence = max(
                        model.predict_proba(
                            transformed_input
                        )[0]
                    ) * 100

            st.divider()

            # Result
            if prediction[0] == 1:

                result = "Real"

                st.success(
                    "✅ The News appears to be REAL."
                )

                st.session_state.real_count += 1

            else:

                result = "Fake"

                st.error(
                    "❌ The News appears to be FAKE."
                )

                st.session_state.fake_count += 1

            # Confidence
            if confidence:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            # Save History
            st.session_state.history.append(
                {
                    "Text": news_input[:60] + "...",
                    "Result": result
                }
            )

        else:
            st.warning(
                "Please enter some news text first."
            )

# -------------------------------------
# HISTORY PAGE
# -------------------------------------
elif page == "History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        for i, item in enumerate(
            st.session_state.history, start=1
        ):

            st.subheader(f"Record {i}")

            st.write(
                f"**News:** {item['Text']}"
            )

            st.write(
                f"**Result:** {item['Result']}"
            )

            st.divider()

        if st.button(
            "🗑️ Clear History",
            help="Remove all prediction history."
        ):

            st.session_state.history = []
            st.session_state.real_count = 0
            st.session_state.fake_count = 0

            st.success(
                "History cleared successfully!"
            )

    else:
        st.info("No predictions available.")

# -------------------------------------
# STATISTICS PAGE
# -------------------------------------
elif page == "Statistics":

    st.title("📊 Analysis Statistics")

    total = (
        st.session_state.real_count
        + st.session_state.fake_count
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Analyzed",
        total
    )

    col2.metric(
        "Real News",
        st.session_state.real_count
    )

    col3.metric(
        "Fake News",
        st.session_state.fake_count
    )

    st.divider()

    if total > 0:

        real_rate = (
            st.session_state.real_count / total
        ) * 100

        fake_rate = (
            st.session_state.fake_count / total
        ) * 100

        st.metric(
            "Real News Rate",
            f"{real_rate:.2f}%"
        )

        st.metric(
            "Fake News Rate",
            f"{fake_rate:.2f}%"
        )

    else:
        st.warning(
            "No news articles have been analyzed yet."
        )

# -------------------------------------
# ABOUT US PAGE
# -------------------------------------
elif page == "About Us":

    st.title("ℹ️ About Us")

    st.write(
        """
        ### Fake News Detector

        Fake News Detector is a Machine Learning
        application developed to identify whether
        a news article is Real or Fake.

        ### Features
        - Fake News Detection
        - Prediction History
        - Analysis Statistics
        - Confidence Score
        - User-Friendly Interface

        ### Technologies Used
        - Python
        - Streamlit
        - Scikit-Learn
        - TF-IDF Vectorizer
        - Logistic Regression
        - Joblib

        ### Developed By
        **Anurag Tripathi**

        ### Version
        1.0.0
        """
    )

    st.success(
        "Thank you for using Fake News Detector!"
    )

# -------------------------------------
# Footer
# -------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "It is for project"
)