import streamlit as st
import json
import os
import time
import random
import sys
import re
from types import SimpleNamespace

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    # llama_cpp may be used with llama-cpp-python if available
    import llama_cpp  # noqa: F401
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HF_HUB_AVAILABLE = False

st.set_page_config(page_title="APAS Chatbot Framework Demo", layout="wide")

# --- Hardcoded Data (Fallback) ----------------------------------------------
HARDCODED_DB = {
    "Sum Function": {
        1: {
            "mcq": {
                "question": "Why is the variable 'sum' initialized to 0 at the start of the function?",
                "options": [
                    "To prevent using a garbage value from memory",
                    "To ensure the loop runs exactly n times",
                    "To allocate memory for the integer",
                    "Because all variables must start at 0 in C"
                ],
                "correct_indices": [0],
                "topic": "Variable Initialization",
                "reference_reason": "Local variables in C are not initialized by default. If not set to 0, 'sum' would contain a random 'garbage' value, making the addition result incorrect."
            },
            "open": {
                "question": "What would happen to the final result if you removed the line 'int sum = 0;'?",
                "reference_answer": "The result would be unpredictable (random) because the accumulator would start with whatever value happened to be at that memory address."
            },
            "keywords": ["garbage", "random", "undefined", "memory", "predictable"]
        },
        2: {
            "mcq": {
                "question": "Why is the loop condition 'i < n' used instead of 'i <= n'?",
                "options": [
                    "To iterate exactly n times (0 to n-1)",
                    "To prevent an infinite loop",
                    "Because arrays are 0-indexed",
                    "To include the value 'n' in the sum"
                ],
                "correct_indices": [0],
                "topic": "Loop Boundaries",
                "reference_reason": "We want to sum 'n' numbers. Starting from 0, the integers are 0, 1, ..., n-1. Using <= would run the loop n+1 times."
            },
            "open": {
                "question": "If n=5, what is the final value of i when the loop terminates?",
                "reference_answer": "The loop terminates when i becomes 5. At that point, 5 < 5 is false, so the loop stops."
            },
            "keywords": ["5", "five", "false", "condition", "terminates", "stops"]
        },
        3: {
            "mcq": {
                "question": "Why uses 'sum += i' inside the loop?",
                "options": [
                    "To accumulate the running total",
                    "To increment the counter",
                    "To assign i to sum",
                    "To check if sum is equal to i"
                ],
                "correct_indices": [0],
                "topic": "Accumulator Logic",
                "reference_reason": "The += operator adds the current value of i to the existing value of sum, creating a cumulative total."
            },
            "open": {
                "question": "Explain the difference between 'sum = i' and 'sum += i'.",
                "reference_answer": "'sum = i' overwrites the previous value, while 'sum += i' adds to it."
            },
            "keywords": ["overwrite", "replace", "add", "plus", "accumulate"]
        }
    },
    "Get Last Element": {
        1: {
            "mcq": {
                "question": "Why is 'size - 1' used to access the array element?",
                "options": [
                    "Because arrays are 0-indexed",
                    "Because the size variable is off by one",
                    "To verify the array isn't empty",
                    "To access the memory address before the array"
                ],
                "correct_indices": [0],
                "topic": "Zero-based Indexing",
                "reference_reason": "In C, arrays start at index 0. Therefore, an array of size N has valid indices from 0 to N-1."
            },
            "open": {
                "question": "What specifically happens if you try to access 'arr[size]'?",
                "reference_answer": "It causes an out-of-bounds access, potentially reading garbage data or causing a segmentation fault."
            },
            "keywords": ["bound", "garbage", "segfault", "crash", "memory", "error"]
        }
    },
    "Binary Search": {
        1: {
            "mcq": {
                "question": "Why is 'mid' calculated as 'left + (right - left) / 2' instead of '(left + right) / 2'?",
                "options": [
                    "To prevent integer overflow for large values of left and right",
                    "Because division by 2 is faster this way",
                    "To ensure the result is always rounded down",
                    "To handle negative numbers correctly"
                ],
                "correct_indices": [0],
                "topic": "Integer Overflow",
                "reference_reason": "If left and right are both large positive integers, adding them might exceed the maximum integer value (overflow) before division. The subtraction method avoids this."
            },
            "open": {
                "question": "Explain in your own words how the standard formula (left+right)/2 could fail.",
                "reference_answer": "If left and right are very large, their sum exceeds 2,147,483,647 (MAX_INT), causing the value to wrap around to a negative number."
            },
            "keywords": ["overflow", "max", "limit", "exceed", "negative", "large"]
        }
    }
}

GENERIC_MCQ = {
    "question": "Why is this specific logic construct used here?",
    "options": [
        "To ensure memory safety and correct execution flow",
        "To optimize the compiler speed",
        "Because it is required by the syntax standard",
        "To handle edge cases properly"
    ],
    "correct_indices": [0, 3],
    "topic": "General Logic (Fallback)",
    "reference_reason": "This code pattern ensures the program behaves correctly across standard inputs and edge cases."
}

GENERIC_OPEN = {
    "question": "Explain the purpose of this block in your own words.",
    "reference_answer": "It implements the core logic needed to solve the problem statement efficiently."
}


# --- Utilities --------------------------------------------------------------
def clear_checkbox_state(prefix: str = "checkbox_") -> None:
    """Remove all checkbox keys from Streamlit session_state."""
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(prefix)]
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state["checkbox_states"] = {}
    st.session_state["is_validating"] = False
    st.session_state["validating_iteration"] = None


# --- Gemini helpers ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_gemini():
    """Configure Gemini client if an API key is available."""
    if not GEMINI_AVAILABLE:
        return None
    # Prioritize environment variable, fallback to secrets, allow blank for now
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    return None


# --- Model Initialization --------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_model(backend_choice, gemini_key=None, hf_token=None):
    """Initialize an available model backend.

    Priority & options:
    - If user selects Gemini and GEMINI_AVAILABLE with API key, use Gemini.
    - If user selects Hugging Face API, use InferenceClient (remote).
    - Else if local transformer model is requested/available, use HF pipeline.
    - Else return None to force fallback logic already present in the app.
    """
    # Helper wrappers to unify API
    class GeminiWrapper:
        def __init__(self, genai_model):
            self._m = genai_model

        def generate_content(self, prompt: str):
            return self._m.generate_content(prompt)

    class HFAPIWrapper:
        def __init__(self, client, model_name):
            self._client = client
            self._model = model_name

        def generate_content(self, prompt: str):
            # Try chat completion first (better for instruction following models)
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self._client.chat_completion(
                    messages=messages,
                    model=self._model,
                    max_tokens=1024,
                    temperature=0.7
                )
                text = response.choices[0].message.content
            except Exception as e:
                # Fallback to simple text generation
                try:
                    text = self._client.text_generation(
                        prompt,
                        model=self._model,
                        max_new_tokens=1024,
                        temperature=0.7
                    )
                except Exception as e2:
                    st.error(f"HF API Error: {e}. Fallback Error: {e2}")
                    text = ""
            return SimpleNamespace(text=text)

    class HFWrapper:
        def __init__(self, model_name):
            from transformers import pipeline
            self._pipe = pipeline("text-generation", model=model_name, max_length=150, num_return_sequences=1)

        def generate_content(self, prompt: str):
            result = self._pipe(prompt)
            return SimpleNamespace(text=result[0]["generated_text"])

    # Try explicit choices first
    if backend_choice == "None (Fallback)":
        return None

    # 1. Gemini
    if backend_choice in ("Auto (Gemini -> HF API -> Local)", "Gemini"):
        if GEMINI_AVAILABLE:
            api_key = gemini_key or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    gm = genai.GenerativeModel("gemini-2.0-flash")
                    return GeminiWrapper(gm)
                except Exception:
                    if backend_choice == "Gemini":
                        return None
        elif backend_choice == "Gemini":
            return None

    # 2. Hugging Face API (Remote, Free Tier)
    if backend_choice in ("Auto (Gemini -> HF API -> Local)", "Hugging Face API"):
        if HF_HUB_AVAILABLE:
            # Get token from env or secrets (optional for some models, but recommended)
            token = hf_token or os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_API_KEY", "")
            # Default to a good instruction tuned model
            # User requested Gemma. Note: Gemma is gated, ensure you have accepted terms on HF website.
            hf_model = os.environ.get("HF_MODEL") or st.secrets.get("HF_MODEL", "google/gemma-2-9b-it")

            try:
                client = InferenceClient(token=token if token else None)
                # Test connection lightly? No, just return wrapper.
                return HFAPIWrapper(client, hf_model)
            except Exception:
                if backend_choice == "Hugging Face API":
                    return None

    # If auto or explicit local requested, try transformers
    if backend_choice in ("Auto (Gemini -> HF API -> Local)", "Local (transformers)"):
        if LLAMA_CPP_AVAILABLE and "llama-cpp-python" in sys.modules:
            # Prefer llama_cpp if available (faster, lower memory)
            model_name = os.environ.get("LLAMA_MODEL", "TheBloke/vicuna-7B-1.1-HF")
            return HFWrapper(model_name)

        try:
            from transformers import pipeline
            model_name = os.environ.get("HF_MODEL", "gpt2")
            pipe = pipeline("text-generation", model=model_name, max_length=150, num_return_sequences=1)
            return HFWrapper(pipe)
        except Exception:
            return None

    return None


def retry_api_call(func, max_retries=3, initial_delay=2):
    """Retry wrapper with exponential backoff for rate limiting errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            # If quota exceeded, don't retry, fail fast to fallback
            if "quota" in message or "429" in message:
                st.warning(f"⚠️ API Quota exceeded. Switching to offline fallback mode.")
                return None

            rate_limited = any(token in message for token in ["rate limit", "too many requests"])
            if rate_limited and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise
    return None


def parse_json_response(response_text: str):
    """Normalize Gemini output into JSON."""
    text = (response_text or "").strip()

    # Attempt to find JSON structure via regex if direct parse fails
    # This helps with chatty models (like Mistral) that add conversational filler
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        if text:
            st.warning(f"⚠️ AI Response was not valid JSON. Raw output:\n{text}")
        return None


def gemini_call(prompt: str, model):
    if not model:
        return None

    def _invoke():
        response = model.generate_content(prompt)
        return parse_json_response(response.text)

    try:
        result = retry_api_call(_invoke)
        if result:
            result["_source"] = "AI Model"  # Mark as AI generated
        return result
    except Exception:
        return None


# --- Prompted tasks (Hybrid) ------------------------------------------------
def get_fallback_data(iteration, key_type):
    selected = st.session_state.selected
    # Get specific example data if available, else generic
    example_data = HARDCODED_DB.get(selected, {}).get(iteration, {})

    if key_type == "mcq":
        return example_data.get("mcq", GENERIC_MCQ)
    elif key_type == "open":
        return example_data.get("open", GENERIC_OPEN)
    elif key_type == "keywords":
        return example_data.get("keywords", [])
    return None


def generate_mcq(code: str, iteration: int, previous_topics: list[str], model):
    # Try API first
    if model:
        topics = ", ".join(topic for topic in previous_topics if topic) or "None"
        prompt = f"""You are an expert programming educator. Generate a multiple-choice question about:
Code: ```{code}```
Iteration: {iteration}/3. Topics covered: {topics}.
Focus on a SPECIFIC line. Output JSON:
{{ "question": "...", "options": ["A","B","C","D"], "correct_indices": [0], "topic": "...", "reference_reason": "..." }}"""
        result = gemini_call(prompt, model)
        if result:
            return result

    # Fallback
    return get_fallback_data(iteration, "mcq")


def generate_open_ended(mcq_data: dict, model):
    # Try API first
    if model:
        prompt = f"""Based on MCQ: {mcq_data['question']} (Answer: {mcq_data['options'][0]}).
Generate a SHORT open-ended 'Why?' question and a 1-sentence reference answer.
Output JSON: {{ "question": "...", "reference_answer": "..." }}"""
        result = gemini_call(prompt, model)
        if result:
            return result

    # Fallback
    return get_fallback_data(st.session_state.current_iteration, "open")


def verify_with_gemini(student_answer: str, reference_reason: str, model):
    # Try API first
    if model:
        prompt = f"""Grade this explanation. Ref: {reference_reason}. Student: {student_answer}.
Output JSON: {{ "similarity_score": <0-100>, "feedback": "...", "hint": "..." }}"""
        result = gemini_call(prompt, model)
        if result:
            return result

    # Fallback Grading Logic (Keyword Matching)
    keywords = get_fallback_data(st.session_state.current_iteration, "keywords")
    result = {}

    if not keywords:
        # Simple length check if no keywords defined
        score = 80 if len(student_answer) > 20 else 40
        result = {"similarity_score": score, "feedback": "Evaluation based on length (Fallback).", "hint": ""}
    else:
        hits = sum(1 for word in keywords if word.lower() in student_answer.lower())

        if hits >= 2:
            result = {"similarity_score": 90, "feedback": "Great job! You hit the key concepts.", "hint": ""}
        elif hits == 1:
            result = {"similarity_score": 60, "feedback": "You're on the right track, but missed some details.",
                    "hint": f"Consider using terms like {random.choice(keywords)}."}
        else:
            result = {"similarity_score": 30, "feedback": "That doesn't seem to match the core concept.",
                    "hint": "Think about the underlying memory or logic mechanics."}

    result["_source"] = "Fallback Logic"
    return result


# --- Demo data --------------------------------------------------------------
EXAMPLES = {
    "Sum Function": {
        "code": """int sum_numbers(int n) {\n    int sum = 0;\n    for (int i = 0; i < n; i++) {\n        sum += i;\n    }\n    return sum;\n}""",
        "unit_test_result": "PASSED (Output: 10 for n=5)",
    },
    "Get Last Element": {
        "code": """int get_last(int arr[], int size) {\n    return arr[size - 1];\n}""",
        "unit_test_result": "PASSED (Returns last element correctly)",
    },
    "Binary Search": {
        "code": """int binary_search(int arr[], int size, int target) {\n    int left = 0;\n    int right = size - 1;\n\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n\n        if (arr[mid] == target)\n            return mid;\n        else if (arr[mid] < target)\n            left = mid + 1;\n        else\n            right = mid - 1;\n    }\n    return -1;\n}""",
        "unit_test_result": "PASSED (Finds elements correctly)",
    },
    "Linked List Reversal": {
        "code": """struct Node* reverse_list(struct Node* head) {\n    struct Node* prev = NULL;\n    struct Node* current = head;\n    struct Node* next = NULL;\n\n    while (current != NULL) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }
    return prev;
}""",
        "unit_test_result": "PASSED (Reverses list correctly)",
    },
}


# --- Session state ----------------------------------------------------------
def bootstrap_state():
    st.session_state.setdefault("selected", list(EXAMPLES.keys())[0])
    st.session_state.setdefault("current_iteration", 1)
    st.session_state.setdefault("iterations", {1: {}, 2: {}, 3: {}})
    st.session_state.setdefault("tier1_done", False)
    st.session_state.setdefault("tier1_correct", False)
    st.session_state.setdefault("demo_mode", False)
    st.session_state.setdefault("show_reference", False)
    st.session_state.setdefault("checkbox_states", {})
    st.session_state.setdefault("is_validating", False)
    st.session_state.setdefault("validating_iteration", None)


bootstrap_state()

# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("Configuration")

backend_choice = st.sidebar.selectbox(
    "Model Backend",
    ("Auto (Gemini -> HF API -> Local)", "Gemini", "Hugging Face API", "Local (transformers)", "None (Fallback)"),
    index=0,
    help="Choose which model backend to use. 'Hugging Face API' runs remotely (free tier available).",
)

# Get or prompt for API keys if Gemini or HF API is selected
gemini_key = None
hf_token = None

if backend_choice == "Gemini" or backend_choice == "Auto (Gemini -> HF API -> Local)":
    default_gemini = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value=default_gemini, help="Required for Gemini model.")

if backend_choice == "Hugging Face API" or backend_choice == "Auto (Gemini -> HF API -> Local)":
    default_hf = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_API_KEY", "")
    hf_token = st.sidebar.text_input("Hugging Face Token", type="password", value=default_hf, help="Optional: Token for Hugging Face Inference API.")

model = init_model(backend_choice, gemini_key, hf_token)

new_demo_mode = st.sidebar.checkbox(
    "Demo Mode",
    value=st.session_state.demo_mode,
    help="Pre-fill the open-ended answer and auto-select the correct MCQ options for quick demos.",
)
if new_demo_mode != st.session_state.demo_mode:
    st.session_state.demo_mode = new_demo_mode
    clear_checkbox_state()
    for iter_state in st.session_state.iterations.values():
        iter_state.pop("demo_seeded", None)

st.sidebar.markdown("---")
choice = st.sidebar.radio("Select Example", list(EXAMPLES.keys()))

if choice != st.session_state.selected:
    st.session_state.selected = choice
    st.session_state.current_iteration = 1
    st.session_state.iterations = {1: {}, 2: {}, 3: {}}
    st.session_state.tier1_done = False
    st.session_state.tier1_correct = False
    st.session_state.show_reference = False
    clear_checkbox_state()

# --- Main layout ------------------------------------------------------------
example = EXAMPLES[st.session_state.selected]
iteration = st.session_state.current_iteration

st.title("APAS Chatbot Assessment Demo")
st.info(f"📍 **Iteration {iteration}/3** - Targeting a new facet of the submission")

col_code, col_tests = st.columns([2, 1])
with col_code:
    st.subheader("Student Submission")
    st.code(example["code"], language="c")
with col_tests:
    st.subheader("Unit Tests")
    st.success(f"✅ {example['unit_test_result']}")
    st.caption("Functional tests passed; now verifying conceptual understanding.")

st.divider()

# --- Tier 1: MCQ ------------------------------------------------------------
iteration_state = st.session_state.iterations[iteration]

if "mcq" not in iteration_state:
    with st.spinner(f"Generating Tier 1 question for iteration {iteration}..."):
        prior_topics = [
            st.session_state.iterations[i].get("mcq", {}).get("topic", "")
            for i in range(1, iteration)
            if "mcq" in st.session_state.iterations[i]
        ]
        mcq = generate_mcq(example["code"], iteration, prior_topics, model)

        # Validation for fallback or API result
        if mcq and isinstance(mcq.get("correct_indices"), list):
            iteration_state["mcq"] = mcq
        else:
            st.warning("⚠️ Using Generic Fallback (API Error or Empty Response)")
            iteration_state["mcq"] = GENERIC_MCQ

mcq = iteration_state["mcq"]
correct_indices = mcq.get("correct_indices", [])

st.subheader(f"Tier 1: Code Understanding (Iteration {iteration})")
st.markdown(f"**Question:** {mcq['question']}")

checkbox_keys = []
for idx, option in enumerate(mcq["options"]):
    checkbox_key = f"checkbox_{st.session_state.selected}_{iteration}_{idx}"
    checkbox_keys.append(checkbox_key)

# Seed demo answers once per iteration if demo mode is active
if st.session_state.demo_mode and not iteration_state.get("demo_seeded", False):
    for idx, key in enumerate(checkbox_keys):
        value = idx in correct_indices
        st.session_state[key] = value
        st.session_state.checkbox_states[key] = value
    iteration_state["demo_seeded"] = True
elif not st.session_state.demo_mode:
    iteration_state["demo_seeded"] = False
    for key in checkbox_keys:
        if key not in st.session_state:
            st.session_state[key] = False
        st.session_state.checkbox_states.setdefault(key, st.session_state[key])

selected_indices = []
if not st.session_state.tier1_done:
    st.write("Select all answers that apply:")
    for idx, option in enumerate(mcq["options"]):
        checkbox_key = checkbox_keys[idx]
        checked = st.checkbox(option, key=checkbox_key)
        st.session_state.checkbox_states[checkbox_key] = checked
        if checked:
            selected_indices.append(idx)

    any_selected = len(selected_indices) > 0

    if st.button("Submit Answer", type="primary", disabled=not any_selected):
        chosen_set = set(idx for idx, key in enumerate(checkbox_keys) if st.session_state.checkbox_states.get(key))
        correct_set = set(correct_indices)
        st.session_state.tier1_correct = chosen_set == correct_set
        st.session_state.tier1_done = True
        st.rerun()
else:
    chosen_set = set(idx for idx, key in enumerate(checkbox_keys) if st.session_state.checkbox_states.get(key))
    correct_set = set(correct_indices)
    if st.session_state.tier1_correct:
        st.success("✅ Perfect selection! Every correct option was chosen and no incorrect options were selected.")
    else:
        st.error("❌ Not quite. Please review the code logic and try again.")

    if st.button("Try Tier 1 Again"):
        st.session_state.tier1_done = False
        st.session_state.tier1_correct = False
        iteration_state.pop("demo_seeded", None)
        clear_checkbox_state()
        st.session_state.show_reference = False
        st.rerun()

# --- Tier 2: Open explanation -----------------------------------------------
if st.session_state.tier1_done and st.session_state.tier1_correct:
    st.divider()
    st.subheader("Tier 2: Explain Your Reasoning")
    st.warning("⚠️ To lock in the score, justify your choice in your own words.")

    if "open_ended" not in iteration_state:
        with st.spinner("Drafting follow-up question..."):
            open_q = generate_open_ended(mcq, model)
            iteration_state["open_ended"] = open_q if open_q else GENERIC_OPEN

    open_q = iteration_state["open_ended"]
    st.info(open_q["question"])

    is_currently_validating = st.session_state.is_validating and st.session_state.validating_iteration == iteration

    default_answer = open_q["reference_answer"] if st.session_state.demo_mode else ""
    prefill_answer = (
            iteration_state.get("pending_explanation")
            or iteration_state.get("explanation")
            or default_answer
    )

    explanation = st.text_area(
        "Your explanation:",
        key=f"explanation_{iteration}",
        value=prefill_answer,
        placeholder="Describe the rationale behind the correct choice...",
        disabled=is_currently_validating,
    )

    col_hint, col_submit = st.columns(2)
    with col_hint:
        if st.button("💡 Show Reference Answer", key=f"toggle_ref_{iteration}", disabled=is_currently_validating):
            st.session_state.show_reference = not st.session_state.show_reference
    with col_submit:
        if st.button("Submit Explanation", type="primary", key=f"submit_{iteration}", disabled=is_currently_validating):
            if explanation.strip():
                iteration_state["pending_explanation"] = explanation.strip()
                iteration_state.pop("tier2_result", None)
                st.session_state.is_validating = True
                st.session_state.validating_iteration = iteration
                st.rerun()
            else:
                st.warning("Please enter an explanation before submitting.")

    if st.session_state.show_reference:
        st.info(f"**Reference Answer:** {open_q['reference_answer']}")

    if is_currently_validating:
        with st.spinner("Verifying your explanation..."):
            explanation_to_grade = iteration_state.get("pending_explanation", "")
            result = verify_with_gemini(explanation_to_grade, open_q["reference_answer"], model)
        iteration_state["tier2_result"] = result
        iteration_state["explanation"] = explanation_to_grade
        iteration_state.pop("pending_explanation", None)
        st.session_state.is_validating = False
        st.session_state.validating_iteration = None

    if "tier2_result" in iteration_state and not st.session_state.is_validating:
        result = iteration_state["tier2_result"]
        tier1_score = 20 if st.session_state.tier1_correct else 0
        tier2_score = result["similarity_score"] * 0.8
        total_score = tier1_score + tier2_score
        iteration_state["score"] = total_score

        # Display Source Indicator
        source = result.get("_source", "Unknown")
        if source == "Fallback Logic":
            st.caption("⚠️ Graded using **Fallback Logic** (Model unavailable or failed to respond correctly).")
        else:
            st.caption(f"🤖 Graded by **{source}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tier 1", f"{tier1_score}/20")
        c2.metric("Tier 2", f"{tier2_score:.0f}/80")
        c3.metric("Iteration Score", f"{total_score:.0f}/100")

        if total_score >= 70:
            st.success(f"🎉 Strong explanation! {result['feedback']}")
        elif total_score >= 50:
            st.warning(f"⚠️ Partial understanding. {result['feedback']}")
        else:
            st.error(f"❌ Needs more clarity. {result['feedback']}")

        if result.get("hint") and total_score < 70:
            st.info(f"💡 Hint: {result['hint']}")

        if iteration < 3:
            if st.button("Continue to Next Iteration →", type="primary"):
                st.session_state.current_iteration += 1
                st.session_state.tier1_done = False
                st.session_state.tier1_correct = False
                st.session_state.show_reference = False
                clear_checkbox_state()
                st.rerun()
        else:
            st.divider()
            st.subheader("🎉 Final Results")
            scores = [st.session_state.iterations[i].get("score", 0) for i in range(1, 4)]
            average = sum(scores) / 3

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Iteration 1", f"{scores[0]:.0f}/100")
            f2.metric("Iteration 2", f"{scores[1]:.0f}/100")
            f3.metric("Iteration 3", f"{scores[2]:.0f}/100")
            f4.metric("Final Grade", f"{average:.0f}/100")

            if average >= 70:
                st.success("🌟 Excellent! You've demonstrated deep code understanding across all tiers.")
            elif average >= 50:
                st.warning("⚠️ Solid attempt. Revisit the weaker concepts and try another round.")
            else:
                st.error("❌ More practice needed. Let's keep exploring those concepts.")

            if st.button("Start Over", type="primary"):
                st.session_state.current_iteration = 1
                st.session_state.iterations = {1: {}, 2: {}, 3: {}}
                st.session_state.tier1_done = False
                st.session_state.tier1_correct = False
                st.session_state.show_reference = False
                clear_checkbox_state()
                st.rerun()

st.divider()
st.caption("🔄 Three-iteration assessment: every pass drills a new conceptual angle.")
