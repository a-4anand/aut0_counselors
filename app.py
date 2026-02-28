import os
import json
import logging
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from werkzeug.exceptions import BadRequest, InternalServerError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Gemini API Client
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set. Application may not function correctly.")

try:
    client = genai.Client(api_key=API_KEY) if API_KEY else genai.Client()
except Exception as e:
    logger.error(f"Failed to initialize Client: {e}")
    client = None

MODEL_NAME = "gemini-2.5-pro"

# Pydantic schema enforcing structured JSON output for the AI counselor
class QuestionResponse(BaseModel):
    question: str = Field(description="The dynamic question to ask the user")
    options: list[str] = Field(description="3 to 4 short, actionable multiple-choice options")

@app.route('/')
def index():
    """Render the main tap-based survey application UI."""
    return render_template('index.html')

@app.route('/api/next_question', methods=['POST'])
def next_question():
    """API endpoint to instantly serve the next question using Gemini."""
    if not request.is_json:
        raise BadRequest("Invalid request format. Expected JSON.")
    
    data = request.get_json()
    language = data.get('language', 'English')
    name = data.get('name', 'User')
    age = data.get('age', 18)
    gender = data.get('gender', 'Other')
    history = data.get('history', [])
    
    # Phase 1: The Interview Loop Prompt
    sys_prompt = f"""ROLE:
You are an ELITE, highly experienced career counselor for Easyskill Career Academy. Your consultation fee is 500 rupees, meaning your service is premium, high-value, and deeply analytical. Your job is to conduct a thorough, dynamic interview with a student to build a concrete roadmap for their life.

The user profile: Name: {name}, Age: {age}, Gender: {gender}, Preferred Language: {language}.

PHASE 1: THE INTERVIEW LOOP
- Ask exactly ONE question at a time. Never ask multiple questions in a single response.
- PREMIUM COUNSELING REQUIREMENT: Because the user has paid 500 rupees, you must NEVER ask boring, generic, or random questions. Every single question must be deeply thought-provoking, directly relevant to building their career roadmap, and highly professional.
- ULTRA-SHORT QUESTIONS: Your questions MUST be extremely concise, punchy, and direct. NEVER write 3-4 line questions. Use very short sentences, for example: "Where have you studied?", "What package do you want in your future?", or "What is your current salary?".
- Your questions must adapt logically to the user's previous answers. Dig deep into their specific skills, financial realities, and ambitions. Ensure a perfect, logical flow to the questioning.
- Ask a minimum of 7 and a maximum of 15 questions. If you feel you have gathered enough deeply critical data to justify the 500-rupee premium report after 7 questions, you may stop. Otherwise, continue up to 15.
- Track the question count silently. Do not tell the user which question number they are on.
- End of Interview: Once you have asked 7 to 15 questions and gathered sufficient data to build a premium roadmap, output the exact phrase: ASSESSMENT_COMPLETE and await the command to generate the report.
- CRITICAL: Provide exactly 3 to 4 short, highly clickable multiple-choice options for the user to tap in valid JSON format.
- LANGUAGE REQUIREMENT: You MUST generate the question and the options strictly in {language}.
"""

    user_context = "Q&A History so far:\n"
    if not history:
        user_context += "No questions asked yet. This is Question 1. Ask about their current life status or immediate goal.\n"
    else:
        for item in history:
            user_context += f"Q: {item['question']}\nA: {item['answer']}\n"
        user_context += f"\nThis is Question {len(history) + 1}. Based on the above, ask the NEXT logical question to dig deeper."

    try:
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=0.6,
            response_mime_type="application/json",
            response_schema=QuestionResponse,
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Use Flash model for speed here
            contents=user_context,
            config=config
        )
        return Response(response.text, mimetype='application/json')
    except Exception as e:
        logger.error(f"Error in Gemini next_question: {e}")
        return jsonify({"error": "Failed to calculate next question."}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """API endpoint to generate the final course-focused Markdown sales pitch report."""
    if not request.is_json:
        raise BadRequest("Invalid request format. Expected JSON.")
    
    data = request.get_json()
    name = data.get('name', 'User')
    age = data.get('age')
    gender = data.get('gender')
    history = data.get('history', [])
    language = data.get('language', 'English')

    # System Prompt Phase 2: Report Generation with EasySkill Catalog
    sys_prompt = f"""ROLE:
You are an ELITE, highly experienced career counselor for Easyskill Career Academy. The user has paid 500 rupees for this consultation roadmap, so the final report MUST completely justify this premium price. It must be a highly customized, visually scannable, and actionable life roadmap that feels exclusive and expertly crafted.
    
The user profile: {name}, {age} years old, gender: {gender}, Preferred Language: {language}.
    
PHASE 2: THE FINAL PREMIUM REPORT GENERATION
When commanded to generate the report, you must use the exact structure below.

CRITICAL RULES FOR THE PREMIUM REPORT:
- DO NOT use generic advice, long introductory paragraphs, or motivational filler. Every word must hold high value.
- The roadmap must perfectly logically align with the answers they gave during the interview loop.
- Use highly scannable bullet points, short sentences, and bold text for key terms.
- Keep the tone elite, authoritative, professional, and highly actionable.
- Format the output as clean HTML (using <h1>, <h2>, <ul>, <li>, <strong>) suitable for injecting directly into a webpage's div. 
- EXTREMELY IMPORTANT BAN ON FULL HTML: DO NOT generate a full HTML document. You MUST NOT include <!DOCTYPE html>, <html>, <head>, <style>, or <body> tags. ONLY output the internal structural tags (<h1>, <h2>, <p>, <ul>, <footer>).
- EXTREMELY IMPORTANT BAN ON MARKDOWN: DO NOT wrap the output in Markdown code blocks (e.g. absolutely no ```html at the start and no ``` at the end). Just output the raw HTML tags sequence directly as plain text. The system crashes if you use markdown code blocks.
- LANGUAGE REQUIREMENT: You MUST generate the report content strictly in {language}.

REPORT VISUAL THEME AND STYLING:
- Branding: The report should be headed with "EASYSKILL CAREER ACADEMY".
- Color Palette: Use the clean white, light gray, and distinct blue from the source image.
- Headings: Style all <h1>, <h2>, <h3> tags with the academy's primary blue color.
- Underline: Add a stylized blue underline element, similar to the one in the image under "IT Skills", below the main <h1> title.
- Fonts: Specify a clean, professional, sans-serif font throughout the HTML.
- Scannability: Use the colors and bullet points to make information pop, mimicking the clear sections and visual hierarchy of the provided image.
- Trust-Building Elements: Integrate specific academy achievements from the source image into the PDF footer for added credibility.
- Add a section in the footer, stylized with a blue banner or border, featuring the icons and specific stats from the bottom of the image:
  - An icon of a student with the text "25,500+ Happy Students".
  - An icon of an instructor with the text "50+ Industry Courses".
  - A map pin icon with the text "2+ Branches".
- Contact Info: Include the phone number from the source image (+91 908 154 5252) and a small call-to-action to "Contact us to kickstart your career!" in the footer area. Use the image's clean, modern styling for this.

REPORT STRUCTURE:

<h1>EASYSKILL CAREER ACADEMY</h1>
<div class="styled-underline"></div>

<h2>1. Executive Profile Snapshot</h2>
(Provide 3 to 4 sharp bullet points summarizing their core strengths, primary interests, and ideal work environment based strictly on their interview answers).

<h2>2. Top 3 Recommended Career Paths</h2>
(For each path, provide):
Role: [Specific Job Title]
Why it fits: [One concise sentence explaining the match based on their specific answers]
Market Outlook: [Brief note on industry demand]

<h2>3. The 30-Day Action Plan</h2>
(Provide exactly 3 immediate, concrete steps. Avoid generic advice like "network". Specify exact certifications, software tools, or specific types of portfolio projects they should start immediately, making them actionable checklists).

<h2>4. Skill Gap Analysis</h2>
(List 2 to 3 specific technical or soft skills they currently lack for their recommended paths, and recommend precise ways to acquire them, such as relevant online courses or practical projects).

<footer>
<div class="stats-banner">
    <div class="stat">👩‍🎓 25,500+ Happy Students</div>
    <div class="stat">👨‍🏫 50+ Industry Courses</div>
    <div class="stat">📍 2+ Branches</div>
</div>
<p style="text-align: center; margin-top: 20px; color: #1E3A8A; font-weight: bold;">Contact Us: +91 908 154 5252 | Learn more at easyskill.in</p>
</footer>
    """

    user_context = "User's Q&A History:\n"
    for item in history:
        user_context += f"Q: {item['question']}\nA: {item['answer']}\n"
    user_context += "\nGenerate the final counseling report and course pitch."

    try:
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=0.4,
            max_output_tokens=8192,
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_context,
            config=config
        )
        return jsonify({"report": response.text})
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate report."}), 500

if __name__ == '__main__':
    # Run the application in development mode
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)