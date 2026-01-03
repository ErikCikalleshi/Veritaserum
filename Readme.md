# APAS Chatbot Assessment Demo

## Overview
This repository hosts a functional prototype designed to bridge the gap between "code correctness" and "code comprehension." In the era of generative AI, traditional auto-graders often fail to detect when students generate working solutions without understanding the underlying logic.

This tool acts as an **Automated Oral Exam**, shifting assessment from black-box testing to interactive dialogue validation.

## The Problem & Solution

**The Problem:** "Unproductive Success." Students can easily generate code that passes unit tests but lack the conceptual knowledge to explain how it works.

**The Solution:** A dual-agent system that interviews the student about their submission:
1.  **Instructor Agent:** Analyzes the code structure and generates specific Multiple Choice Questions (Tier 1).
2.  **Verifier Agent:** Grades the student's natural language explanations against a hidden ground truth (Tier 2).

If a student struggles, the system provides scaffolded hints rather than revealing the solution.

## Quick Start

**1. Clone and Install**
```bash
git clone [https://github.com/YourUsername/apas-chatbot-assessment.git](https://github.com/YourUsername/apas-chatbot-assessment.git)
cd apas-chatbot-assessment
pip install streamlit google-generativeai huggingface_hub
```
2. Configure AI (Optional) The application includes a "Fallback Mode" that works without API keys. To use the live AI features, set your API key as an environment variable:
```bash
# Linux/Mac
export GEMINI_API_KEY="your_key_here"
# Optional: For Hugging Face
export HF_TOKEN="your_hf_token_here"
export HF_MODEL="google/gemma-2-9b-it" # Optional: Custom model (e.g., Gemma is faster)

# Windows (Powershell)
$env:GEMINI_API_KEY="your_key_here"
# Optional: For Hugging Face
$env:HF_TOKEN="your_hf_token_here"
$env:HF_MODEL="google/gemma-2-9b-it" # Optional: Custom model
```
3. Run the Application
```bash
streamlit run app.py
```

## Technical Approach

This prototype implements the specific architecture proposed in the research paper:

* Two-Tier Scoring: The system uses a weighted formula where 80% of the score depends on the written explanation, ensuring that guessing a multiple-choice answer is not enough to pass. 
$Formula: Score = 20 + (Semantic Similarity * 0.8)$
* Trace-and-Verify Workflow: The assessment iterates through three distinct logical components of the code (e.g., Loop Boundaries, Memory Allocation) to ensure comprehensive coverage.
* Model Agnostic: The code supports Google Gemini 2.0 Flash, Hugging Face Inference API, and Local Transformers.

## Limitations
This is a Proof of Concept. It relies on pre-loaded C code examples (e.g., Binary Search, Linked Lists) to demonstrate the assessment flow described in the literature. It does not currently support dynamic file uploads or live compilation.