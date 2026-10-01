
import streamlit as st
import pandas as pd
from portable import Model

st.title("Study Office – Student Risk")

# Load model
model = Model("model")

# Load data
URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

new = pd.read_csv(URL + "new_week6.csv")
history = pd.read_csv(URL + "history_week6.csv")

# -------------------------
# 1. THIS WEEK'S LIST
# -------------------------

new["risk"] = model.predict_proba(new)

ranked = new.sort_values("risk", ascending=False).copy()
ranked["top_40"] = False
ranked.loc[ranked.index[:40], "top_40"] = True

st.header("This week's list")

st.write(
    "Students are ranked from highest to lowest predicted risk. "
    "The top 40 are marked for contact."
)

st.dataframe(
    ranked[
        [
            "student_id",
            "risk",
            "international",
            "logins_total",
            "weeks_since_login",
            "top_40",
        ]
    ]
)

# -------------------------
# 2. THE MISTAKES OF A RULE
# -------------------------

st.header("The mistakes of a rule")

val = history[history["cohort"] == 2025].copy()

# Predict risk for 2025 students
val["risk"] = model.predict_proba(val)

# Slider
cutoff = st.slider(
    "Risk cut-off",
    min_value=0.05,
    max_value=0.50,
    value=0.50,
    step=0.05,
)

truth = val["left"].values
risk = val["risk"].values

contacted = risk >= cutoff

TP = int((contacted & (truth == 1)).sum())
FP = int((contacted & (truth == 0)).sum())
FN = int((~contacted & (truth == 1)).sum())
TN = int((~contacted & (truth == 0)).sum())

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

st.subheader("What happens with this rule?")

st.write(f"{TP} students were reached in time.")
st.write(f"{FP} students were worried for nothing.")
st.write(f"{FN} students who left were missed.")
st.write(f"{TN} students stayed and were not contacted.")

st.write(f"**Precision:** {precision:.0%}")
st.write(f"**Recall:** {recall:.0%}")
# 3. Per group
st.header("3. Domestic vs International students")

for group, name in [(0, "Domestic"), (1, "International")]:
    g = val[val["international"] == group]

    truth_g = g["left"].values
    risk_g = g["risk"].values
    contacted_g = risk_g >= cutoff

    TP_g = int((contacted_g & (truth_g == 1)).sum())
    FP_g = int((contacted_g & (truth_g == 0)).sum())
    FN_g = int((~contacted_g & (truth_g == 1)).sum())
    TN_g = int((~contacted_g & (truth_g == 0)).sum())

    precision_g = TP_g / (TP_g + FP_g) if (TP_g + FP_g) > 0 else 0
    recall_g = TP_g / (TP_g + FN_g) if (TP_g + FN_g) > 0 else 0

    st.subheader(name)

    st.write(f"Students reached in time: **{TP_g}**")
    st.write(f"Students worried for nothing: **{FP_g}**")
    st.write(f"Students missed: **{FN_g}**")
    st.write(f"Students correctly not contacted: **{TN_g}**")

    st.write(f"**Precision:** {precision_g:.0%}")
    st.write(f"**Recall:** {recall_g:.0%}")
    # 4. Decision support
st.header("4. Decision support")

number_contacted = int((risk >= cutoff).sum())

st.write(
    f"At a {cutoff:.0%} cut-off, the study office would contact "
    f"**{number_contacted} students**."
)

st.write(
    "A lower cut-off means contacting more students, while a higher "
    "cut-off means contacting fewer students."
)
