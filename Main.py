import os
import ast
import json
import time
import streamlit as st

BIAS_PATH = "Bias"

def load_bias():
    if not os.path.exists(BIAS_PATH):
        return [0.0] * 20

    try:
        with open(BIAS_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return [0.0] * 20

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
            return [float(x) for x in parsed]

    except Exception:
        pass

    return [0.0] * 20

def save_bias(bias_list):
    with open(BIAS_PATH, "w", encoding="utf-8") as f:
        json.dump([float(x) for x in bias_list], f)

def neuron(ni1, ni2=0.0, bias=0.0, acceptance=2.0):
    if ni1 + ni2 + bias >= acceptance:
        return ni1 + ni2
    return 0.0

def usv(signal, bias_value, updater):
    if signal != 0:
        return bias_value + updater
    return bias_value - updater

def get_max(var):
    if not var:
        return 0.0

    current = var[0]
    for num in var[1:]:
        if num > current:
            current = num
    return current

def clean_numeric_input(text):
    if text is None:
        return 0.0

    s = text.lower()
    replacements = {
        "hi": "1",
        "hello": "1",
        "?": "1",
        "how are you": "2",
        "good": "3",
        "can you": "4",
        "what": "5",
        "do": "6",
        "is": "7",
        "are you": "8",
        "a": "9",
        "day": "10",
        "time": "11",
        "computer": "12",
        "in": "13",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = s.replace(" ", "0")
    s = s.replace("?", "")
    s = s.replace("!", "")
    s = s.replace(".", "")
    s = s.replace(",", "")
    s = s.replace("-", "")

    try:
        return float(s)
    except ValueError:
        return 0.0

st.title("Talk to ANN — Your personal AI assistant (v1.0)")

if "bias" not in st.session_state:
    st.session_state.bias = load_bias()

user_input = st.text_input("Type in anything:", placeholder="Ask me anything...")

if st.button("Submit Question"):
    if not user_input or not user_input.strip():
        st.warning("Please enter a message before submitting.")
        st.stop()

    B = st.session_state.bias
    if len(B) < 20:
        B = [0.0] * 20

    try:
        Input = clean_numeric_input(user_input)

        # Input layer
        N1 = neuron(Input, acceptance=0.5)
        N2 = neuron(Input, acceptance=1.0)
        N3 = neuron(Input, acceptance=1.5)
        N4 = neuron(Input, acceptance=2.0)
        N5 = neuron(Input, acceptance=2.5)
        N6 = neuron(Input, acceptance=3.0)
        N7 = neuron(Input, acceptance=3.5)
        N8 = neuron(Input, acceptance=4.0)

        # Hidden layer 1
        N9 = neuron(N1, N2, bias=B[1], acceptance=1.5)
        B[1] = usv(N9, B[1], 0.5)

        N10 = neuron(N2, N3, bias=B[2], acceptance=4.5)
        B[2] = usv(N10, B[2], 0.5)

        N11 = neuron(N3, N4, bias=B[3], acceptance=6.5)
        B[3] = usv(N11, B[3], 0.5)

        N12 = neuron(N4, N5, bias=B[4], acceptance=7.5)
        B[4] = usv(N12, B[4], 0.5)

        N13 = neuron(N5, N6, bias=B[5], acceptance=9.5)
        B[5] = usv(N13, B[5], 0.5)

        N14 = neuron(N6, N7, bias=B[6], acceptance=10.0)
        B[6] = usv(N14, B[6], 0.5)

        N15 = neuron(N7, N8, bias=B[7], acceptance=12.5)
        B[7] = usv(N15, B[7], 0.5)

        # Separate hidden layer
        N16 = neuron(N1, N2, B[8], 1.0)
        B[8] = usv(N16, B[8], 0.5)

        NE1 = neuron(N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8, acceptance=50.0)

        # Hidden layer 2
        N17 = neuron(N9, N10, bias=B[9], acceptance=8.0)
        B[9] = usv(N17, B[9], 0.5)

        N18 = neuron(N11, N12, bias=B[10], acceptance=8.5)
        B[10] = usv(N18, B[10], 0.5)

        N19 = neuron(N13, N14, bias=B[11], acceptance=9.5)
        B[11] = usv(N19, B[11], 0.5)

        N20 = neuron(N14, N15, bias=B[12], acceptance=10.0)
        B[12] = usv(N20, B[12], 0.5)

        # Hidden layer 3
        N21 = neuron(N17, N18, bias=B[13], acceptance=11.0)
        B[13] = usv(N21, B[13], 0.5)

        N22 = neuron(N20, N19, bias=B[14], acceptance=12.5)
        B[14] = usv(N22, B[14], 0.5)

        # Output
        NO1 = neuron(N21, N22, bias=B[15], acceptance=15.5)
        NO2 = neuron(N16, N17, bias=B[16], acceptance=1.0)

        # Input-driven adaptive bias
        if NO1 != 0:
            B[15] += 0.02
        else:
            B[15] -= 0.01

        if NO2 != 0:
            B[16] += 0.02
        else:
            B[16] -= 0.01

        CO = get_max([NO1, NO2])
        if Input != 0 and NE1 != 0:
            CO -= NE1 / (Input / 50.0)

        FO = str(CO).replace("e+", "").replace(".", "").replace("0", " ").replace("1", "Hello! ").replace("2", "Hi!").replace("-64", "Good! How are you? ").replace("Good! How are you? Good! How are you? ", "Good! How are you? ").replace("8 64576", "I can sometimes engage in normal everyday conversation. ").replace("8 6 96", "I can sometimes engage in normal everyday conversation. ").replace("64576", "I can ").replace("6 96", "I can ").replace("Hello! Hi!8Hi! 94379Hi!", "Yes I am in fact running in your computer right now!").replace("93376", "").replace("Hello! Hi!8Hi!Hi!Hi!4Hello! ", "Yes I am in fact running in your computer right now!").replace("536", "").replace("8976", "").replace("Hello! Hi!8Hi! 944Hello!  ", "Yes I am in fact running in your computer right now!").replace("Hello! 6I can sometimes engage in normal everyday conversation.", "Hello! I can sometimes engage in normal everyday conversation.")

        if not FO.strip():
            FO = 'A response from ANN was not generated please try again.'

        st.write("An AI Response Has Been Generated")

        output_placeholder = st.empty()
        displayed_text = ""

        for word in FO.split():
            displayed_text += word + " "
            output_placeholder.markdown(displayed_text)
            time.sleep(0.15)

        st.session_state.bias = B
        save_bias(B)

    except Exception as e:
        st.error("Something went wrong while processing your message.")
        st.exception(e)
