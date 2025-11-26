## Veritaserum: Chatbot-Based Code Assessment Demo

This repository hosts a Streamlit application demonstrating a GenAI-powered assessment framework for programming education. Designed to bridge the gap between "code correctness" and "code comprehension," this tool verifies that students truly understand the code they submit, preventing "unproductive success" where students generate working solutions without conceptual depth.

How it Works: The system uses Google Gemini to drive a 3-iteration Socratic assessment process:

* Tier 1 (Knowledge Check): The "Instructor Agent" analyzes the code and generates specific Multiple Choice Questions (MCQs) targeting distinct constructs (e.g., pointers, recursion, edge cases).
* Tier 2 (Reasoning Verification): Students must explain their logic in natural language. The "Verifier Agent" grades this explanation against a hidden reference reason, providing scaffolded hints if the student shows partial understanding.

Key Features:
* Dual-Agent Architecture: Separates question generation from answer verification to reduce hallucinations.
* Two-Tier Scoring: Combines automated selection accuracy with semantic analysis of open-ended explanations.
* State Tracking: Ensures the assessment drills down into different topics over three distinct iterations.

Demo Mode: Pre-configured states for rapid testing and demonstration.