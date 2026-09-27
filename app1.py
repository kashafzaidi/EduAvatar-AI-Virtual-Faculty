import os
import streamlit as st
import fitz
import faiss
import numpy as np
import json
from google import genai
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import base64
def create_local_topics(chunks):
    """
    Gemini unavailable hone par PDF chunks se
    basic lesson topics create karta hai.
    """

    if not chunks:
        return []

    topic_count = min(5, len(chunks))

    group_size = max(1, len(chunks) // topic_count)

    topics = []

    for i in range(topic_count):
        start = i * group_size
        end = len(chunks) if i == topic_count - 1 else (i + 1) * group_size

        group_text = " ".join(chunks[start:end])

        words = group_text.replace("\n", " ").split()

        # First few meaningful words as topic title
        title = " ".join(words[:8])

        if not title:
            title = f"PDF Topic {i + 1}"

        topics.append(f"{i + 1}. {title}")

    return topics



def create_local_faculty_explanation(topic, pdf_content, language):

    if language == "Hindi":
        return f"""
Namaste students.

Aaj hum {topic} ko samjhenge.

Aapke PDF ke according, is topic se related important material ye hai:

{pdf_content[:450]}

Is topic ke main concepts ko dhyan se samjhiye.
Agar koi doubt ho, aap AI faculty se question pooch sakte hain.
"""

    elif language == "Hinglish":
        return f"""
Hello students.

Aaj hum {topic} ko simple way mein samjhenge.

Aapke PDF mein is topic se related important material ye hai:

{pdf_content[:450]}

Iske main concepts ko carefully samjho aur important points note karo.
Agar koi doubt ho, AI faculty se question pooch sakte ho.
"""

    else:
        return f"""
Hello students.

Today we are going to learn about {topic}.

According to your PDF, the important study material is:

{pdf_content[:450]}

Focus on the main concepts and important points.
If you have any doubt, you can ask the AI faculty.
"""



# GEMINI CLIENT


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)



os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# PAGE CONFIG


st.set_page_config(
    page_title="EduAvatar",
    page_icon="🎓",
    layout="wide"
)



# SESSION STATE


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False



# CUSTOM CSS


st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e1b4b 50%,
            #312e81 100%
        );
    }


    /* MAIN HEADINGS */

    h1, h2 {
        color: #38bdf8 !important;
    }


    /* EDUAVATAR TITLE */

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-top: 40px;
        margin-bottom: 10px;
    }


    /* SUBTITLE */

    .subtitle {
        text-align: center;
        font-size: 18px;
        font-weight: 500;
        color: #fbbf24 !important;
        margin-bottom: 35px;
    }


    /* SMALL HEADINGS */

    h3, h4, h5, h6 {
        color: #fbbf24 !important;
    }


    /* NORMAL TEXT */

    .stApp p {
        color: #bae6fd !important;
    }


    /* FORM LABELS */

    label {
        color: #bae6fd !important;
    }


    /* RADIO BUTTONS */

    div[role="radiogroup"] label {
        color: #bae6fd !important;
    }


    /* LOGIN BOX */

    .login-box {
        max-width: 430px;
        margin: auto;
        padding: 35px;
        background: rgba(15, 23, 42, 0.90);
        border: 1px solid rgba(56, 189, 248, 0.30);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }


    /* LOGIN TEXT */

    .login-box h1,
    .login-box h2,
    .login-box h3 {
        color: #38bdf8 !important;
    }

    .login-box label {
        color: #bae6fd !important;
    }


    /* INPUT BOX */

    .login-box input {
        color: white !important;
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(56,189,248,0.30) !important;
        border-radius: 10px !important;
    }

    .login-box input::placeholder {
        color: #94a3b8 !important;
    }


    /* HIDE STREAMLIT BRANDING */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }



    /* ============================== */
/* SIDEBAR */
/* ============================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #071a3d 0%,
        #0b2852 50%,
        #071a3d 100%
    ) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

[data-testid="stSidebarContent"] {
    background: transparent !important;
}

/* Sidebar navigation text */

section[data-testid="stSidebar"] label {
    color: #bae6fd !important;
}

/* Navigation hover */

section[data-testid="stSidebar"] label:hover {
    color: #ffffff !important;
}

/* Radio group */

section[data-testid="stSidebar"] [role="radiogroup"] {
    background: transparent !important;
}

/* Individual menu items */

section[data-testid="stSidebar"] [role="radio"] {
    background: transparent !important;
}

/* Selected menu item */

section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: rgba(56, 189, 248, 0.20) !important;
    border-radius: 10px !important;
}

/* Sidebar headings/text */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #bae6fd !important;
}

/* Sidebar divider */

section[data-testid="stSidebar"] hr {
    border-color: rgba(186, 230, 253, 0.20) !important;
}
     /* ==============================
       HOME PAGE
       ============================== */

    .hero-section {
        text-align: center;
        padding: 35px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #071a33, #123b63);
        margin-bottom: 20px;
    }

    .hero-section h1 {
        color: #38BDF8 !important;
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero-section p {
        color: #FFD166 !important;
        font-size: 20px;
    }


    /* FEATURE CARDS */

    .feature-card {
        background: linear-gradient(145deg, #0b2a4a, #102f50);
        padding: 25px;
        border-radius: 18px;
        min-height: 190px;
        text-align: center;
        border: 1px solid #214f73;
    }

    .feature-card h2 {
        font-size: 38px;
        margin-bottom: 5px;
    }

    .feature-card h3 {
        color: #FFD166 !important;
    }

    .feature-card p {
        color: #87CEEB !important;
        line-height: 1.6;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOGIN / SIGNUP PAGE
# =====================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="main-title">🎓 EduAvatar</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your AI Virtual Faculty for Smarter Learning'
        '</div>',
        unsafe_allow_html=True
    )


    option = st.radio(
        "",
        [
            "Login",
            "Create Account"
        ],
        horizontal=True
    )


    


 
    # LOGIN
  

    if option == "Login":

        st.subheader("Welcome Back 👋")

        email = st.text_input("Email")

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.button(
            "Login",
            use_container_width=True
        )


        if login:

            if email and password:

                st.session_state.logged_in = True

                st.rerun()

            else:

                st.error(
                    "Please enter email and password."
                )


  
    # SIGNUP
  

    else:

        st.subheader("Create Your Account ")

        name = st.text_input("Full Name")

        email = st.text_input("Email")

        password = st.text_input(
            "Create Password",
            type="password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password"
        )

        signup = st.button(
            "Create Account",
            use_container_width=True
        )


        if signup:

            if not name or not email or not password:

                st.error(
                    "Please fill all fields."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match!"
                )

            else:

                st.success(
                    "Account created successfully! "
                    "You can now login."
                )


    
   
# DASHBOARD


else:

    
    # SIDEBAR
   

    with st.sidebar:

        st.markdown(
            "# 🎓 EduAvatar"
        )

        st.markdown(
            "### AI Virtual Faculty"
        )

        st.divider()

        menu = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📚 My PDFs",
                "👨‍🏫 AI Faculty",
                "🎭 Virtual Classroom",
                "📝 Quiz",
                "📊 Analytics",
                "⚙️ Settings"
            ]
        )

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.rerun()


  
    # HOME
  

    if menu == "🏠 Home":

        st.markdown("""
        <div class="hero-section">
            <h1>🎓 Welcome to EduAvatar</h1>
            <p>Your AI Virtual Faculty for Smarter Learning</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h2>📚</h2>
                <h3>Upload Notes</h3>
                <p>Upload your study PDFs and let EduAvatar understand your learning material.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h2>👨‍🏫</h2>
                <h3>AI Faculty</h3>
                <p>Learn from an AI faculty that explains concepts in a simple and interactive way.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <h2>📝</h2>
                <h3>Smart Quiz</h3>
                <p>Test your understanding with AI-powered quizzes based on your study material.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if "pdf_name" in st.session_state:
            st.success(f"📄 Current study material: {st.session_state.pdf_name}")
        else:
            st.info(
                "💡 Start your learning journey by uploading a PDF from the "
                "'My PDFs' section."
            )


   
    # MY PDFs - ONLY PDF PROCESSING / KNOWLEDGE BASE
  

    elif menu == "📚 My PDFs":

        st.markdown("""
        <div class="hero-section">
            <h1>📚 My PDFs</h1>
            <p>Upload your study material and create an AI knowledge base</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📄 Upload your study PDF",
            type=["pdf"]
        )

        if uploaded_file is not None:

            pdf_bytes = uploaded_file.getvalue()
            pdf_signature = f"{uploaded_file.name}_{len(pdf_bytes)}"

            # Process only when a new PDF is uploaded.
            if st.session_state.get("pdf_signature") != pdf_signature:

                st.session_state.pop("topics", None)
                st.session_state.pop("current_topic", None)
                st.session_state.pop("quiz_questions", None)
                st.session_state.pop("quiz_topic", None)
                st.session_state.pop("quiz_submitted", None)

                # Remove lessons generated for the previous PDF.
                for key in list(st.session_state.keys()):
                    if str(key).startswith("lesson_"):
                        del st.session_state[key]

                with st.spinner("📖 Reading your PDF..."):

                    pdf_document = fitz.open(
                        stream=pdf_bytes,
                        filetype="pdf"
                    )

                    full_text = ""

                    for page in pdf_document:
                        full_text += page.get_text()

                    pdf_document.close()

                if not full_text.strip():
                    st.error("❌ No readable text was found in this PDF.")
                    st.stop()

                st.session_state.full_text = full_text
                st.session_state.pdf_name = uploaded_file.name

                st.success("📖 PDF text extracted successfully!")

              
                # TEXT CHUNKING
               

                chunk_size = 1000
                overlap = 200
                chunks = []
                start_pos = 0

                while start_pos < len(full_text):

                    end_pos = start_pos + chunk_size
                    chunk = full_text[start_pos:end_pos]

                    if chunk.strip():
                        chunks.append(chunk)

                    start_pos += chunk_size - overlap

                st.session_state.chunks = chunks
                st.success(f"🧩 Created {len(chunks)} text chunks.")

                
                # GEMINI EMBEDDINGS
                

                with st.spinner("🧠 Creating embeddings..."):

                    result = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=chunks
                    )

                    embeddings = np.array(
                        [item.values for item in result.embeddings],
                        dtype="float32"
                    )

                st.session_state.embeddings = embeddings
                st.success("✅ Embeddings created successfully!")

              
                # FAISS VECTOR DATABASE
                

                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                st.session_state.faiss_index = index

                st.success(
                    f"🗂️ FAISS database created with {index.ntotal} vectors."
                )

                
                # IDENTIFY MAIN TOPICS
               

                with st.spinner("👨‍🏫 Faculty is studying your PDF..."):

                    topic_prompt = f"""
You are an AI college faculty member.

Read the study material below and identify the MAIN topics that should be taught to a student.

IMPORTANT:
- Give only 4 to 6 major topics.
- Do NOT create tiny subtopics.
- Do NOT explain the topics yet.
- Return only a numbered list of topic names.
- Use the terminology from the PDF.

Study Material:
{full_text}
"""

                    try:
                        topic_response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=topic_prompt
                        )

                        st.session_state.topics = topic_response.text
                        st.session_state.current_topic = 0

                    except Exception:
                        st.info(
                            "🔄 Gemini unavailable. Using local PDF Faculty mode."
                        )

                        local_topics = create_local_topics(
                            st.session_state.chunks
                        )

                        st.session_state.topics = "\n".join(
                            local_topics
                        )

                        st.session_state.current_topic = 0

                st.session_state.pdf_signature = pdf_signature

                st.session_state.pdf_signature = pdf_signature

            else:
                st.success(f"📄 {st.session_state.pdf_name} is already loaded.")

            # If topic generation failed because Gemini was busy, retry it here
            # without rebuilding the embeddings/FAISS database.
            if "topics" not in st.session_state:

                with st.spinner("👨‍🏫 Faculty is identifying the main topics..."):

                    topic_prompt = f"""
You are an AI college faculty member.

Read the study material below and identify the MAIN topics that should be taught to a student.

IMPORTANT:
- Give only 4 to 6 major topics.
- Do NOT create tiny subtopics.
- Do NOT explain the topics yet.
- Return only a numbered list of topic names.
- Use the terminology from the PDF.

Study Material:
{st.session_state.full_text}
"""

                    try:
                        topic_response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=topic_prompt
                        )
                        st.session_state.topics = topic_response.text
                        st.session_state.current_topic = 0
                        st.success("🎉 Your PDF knowledge base is ready!")
                        st.info("👉 Open **AI Faculty** to start learning topic by topic.")
                    except Exception:
                        st.warning(
                            "⚠️ Gemini is temporarily busy while identifying topics. "
                            "Please refresh and try again. Your PDF knowledge base is already saved."
                        )
            else:
                st.success("🎉 Your PDF knowledge base is ready!")

            # Preview only. No teaching happens here.
            with st.expander("📄 View extracted text"):
                st.write(st.session_state.full_text)

            with st.expander("🧩 View first 5 PDF chunks"):
                for i, chunk in enumerate(st.session_state.chunks[:5]):
                    st.markdown(f"### Chunk {i + 1}")
                    st.write(chunk)

            if "topics" in st.session_state:
                st.markdown("### 📚 Detected Lesson Topics")
                st.info(st.session_state.topics)


  
    # AI FACULTY - TOPIC-WISE TEACHING
   

    elif menu == "👨‍🏫 AI Faculty":

        if "chunks" not in st.session_state:
            st.warning("📚 Please upload a PDF first from **My PDFs**.")
            st.stop()

        if "topics" not in st.session_state:
            st.warning(
                "⚠️ Topics are not available yet. "
                "Please upload your PDF again from **My PDFs**."
            )
            st.stop()

        st.markdown("## 👨‍🏫 AI Virtual Faculty")
        
# FULL LECTURE HALL + HUMAN FACULTY


        
       
        # SELECTED LANGUAGE
       

        selected_language = st.session_state.get(
            "selected_language",
            "English"
        )

        st.info(
            f"🌐 Faculty Language: **{selected_language}**"
        )

       # LESSON PLAN
      

        st.markdown("### 📚 Today's Lesson Plan")

        st.info(st.session_state.topics)

        topic_lines = [
            line.strip()
            for line in st.session_state.topics.split("\n")
            if line.strip()
        ]

        if not topic_lines:
            st.warning("No topics were detected in the PDF.")
            st.stop()

     
        # CURRENT TOPIC
      

        current_index = (
            st.session_state.get("current_topic", 0)
            % len(topic_lines)
        )

        current_topic = topic_lines[current_index]

        st.markdown("---")

        st.markdown(
            f"## 👨‍🏫 {current_topic}"
        )



        st.components.v1.html(
            f"""
            <div class="lecture-room">

                <!-- DIGITAL BOARD -->
                <div class="board">
                    <div class="board-title">
                        EDUAVATAR • AI VIRTUAL FACULTY
                    </div>

                    <div class="board-topic">
                        {current_topic}
                    </div>

                    <div class="board-line"></div>

                    <div class="board-text">
                        <span>Today's Topic</span>
                        <br><br>
                        Let's understand this concept
                        step by step.
                    </div>
                </div>


                <!-- FACULTY -->
                <div class="faculty-walk">

                    <div class="faculty">

                        <!-- HEAD -->
                        <div class="head">
                            <div class="hair"></div>

                            <div class="face">

                                <div class="eye left-eye"></div>
                                <div class="eye right-eye"></div>

                                <div class="nose"></div>

                                <div class="mouth"></div>

                            </div>
                        </div>


                        <!-- BODY -->
                        <div class="body">

                            <div class="shirt"></div>

                            <!-- LEFT ARM -->
                            <div class="arm left-arm">
                                <div class="hand"></div>
                            </div>

                            <!-- RIGHT ARM -->
                            <div class="arm right-arm">
                                <div class="hand"></div>
                            </div>

                        </div>


                        <!-- LEGS -->
                        <div class="legs">

                            <div class="leg left-leg">
                                <div class="shoe"></div>
                            </div>

                            <div class="leg right-leg">
                                <div class="shoe"></div>
                            </div>

                        </div>

                    </div>

                </div>


                <!-- PODIUM -->
                <div class="podium">
                    <div class="laptop"></div>
                </div>


                <!-- STUDENT DESKS -->
                <div class="desk desk-one"></div>
                <div class="desk desk-two"></div>
                <div class="desk desk-three"></div>


                <!-- STATUS -->
                <div class="faculty-status">
                    👩‍🏫 Faculty is teaching
                </div>

            </div>


            <style>

            * {{
                box-sizing:border-box;
            }}

            .lecture-room {{
                position:relative;
                width:100%;
                height:620px;
                overflow:hidden;
                border-radius:20px;

                background:
                    linear-gradient(
                        rgba(8,20,40,0.25),
                        rgba(8,20,40,0.55)
                    ),
                    linear-gradient(
                        135deg,
                        #152238,
                        #243b55
                    );

                border:1px solid rgba(79,195,247,0.35);

                box-shadow:
                    0 20px 50px rgba(0,0,0,0.35);
            }}


            /* CLASSROOM WALL */

            .lecture-room::before {{
                content:"";
                position:absolute;
                left:0;
                right:0;
                top:0;
                height:58%;

                background:
                    linear-gradient(
                        90deg,
                        #162338,
                        #263d55,
                        #162338
                    );

                border-bottom:8px solid #8a6a3a;
            }}


            /* BOARD */

            .board {{
                position:absolute;

                top:45px;
                left:8%;
                width:84%;
                height:280px;

                padding:28px 35px;

                background:
                    linear-gradient(
                        135deg,
                        #071521,
                        #102b3e
                    );

                border:8px solid #604b32;

                border-radius:8px;

                box-shadow:
                    0 12px 30px rgba(0,0,0,0.45);

                color:#e8f7ff;

                z-index:2;
            }}


            .board-title {{
                font-size:16px;
                letter-spacing:2px;
                color:#62d7ff;
                font-weight:bold;
            }}


            .board-topic {{
                margin-top:25px;

                font-size:32px;
                font-weight:bold;

                color:#ffffff;
            }}


            .board-line {{
                width:180px;
                height:4px;

                margin-top:10px;

                background:#ffd54f;
            }}


            .board-text {{
                margin-top:22px;

                font-size:18px;
                line-height:1.6;

                color:#ccecff;
            }}


            .board-text span {{
                color:#ffd54f;
                font-size:20px;
                font-weight:bold;
            }}


            /* FACULTY WALKING AREA */

            .faculty-walk {{
                position:absolute;

                left:7%;
                bottom:115px;

                width:70%;
                height:250px;

                z-index:8;

                animation:
                    facultyWalk
                    14s
                    ease-in-out
                    infinite;
            }}


            @keyframes facultyWalk {{

                0% {{
                    transform:translateX(0);
                }}

                25% {{
                    transform:translateX(35%);
                }}

                50% {{
                    transform:translateX(65%);
                }}

                75% {{
                    transform:translateX(30%);
                }}

                100% {{
                    transform:translateX(0);
                }}

            }}


            /* HUMAN BODY */

            .faculty {{
                position:absolute;

                bottom:0;
                left:35%;

                width:130px;
                height:235px;

                animation:
                    bodyMove
                    1.2s
                    ease-in-out
                    infinite;
            }}


            @keyframes bodyMove {{

                0% {{
                    transform:translateY(0);
                }}

                50% {{
                    transform:translateY(-4px);
                }}

                100% {{
                    transform:translateY(0);
                }}

            }}


            /* HEAD */

            .head {{
                position:absolute;

                top:0;
                left:38px;

                width:72px;
                height:82px;

                border-radius:50%;

                background:#d69b72;

                z-index:5;

                animation:
                    headMove
                    2.5s
                    ease-in-out
                    infinite;
            }}


            @keyframes headMove {{

                0% {{
                    transform:rotate(0deg);
                }}

                45% {{
                    transform:rotate(-3deg);
                }}

                70% {{
                    transform:rotate(3deg);
                }}

                100% {{
                    transform:rotate(0deg);
                }}

            }}


            /* HAIR */

            .hair {{
                position:absolute;

                top:-8px;
                left:5px;

                width:62px;
                height:34px;

                border-radius:
                    50% 50% 30% 30%;

                background:#241a18;
            }}


            /* FACE */

            .face {{
                position:absolute;

                left:0;
                top:18px;

                width:100%;
                height:65px;
            }}


            .eye {{
                position:absolute;

                top:22px;

                width:7px;
                height:7px;

                border-radius:50%;

                background:#222;
            }}


            .left-eye {{
                left:20px;
            }}


            .right-eye {{
                right:20px;
            }}


            .nose {{
                position:absolute;

                left:33px;
                top:31px;

                width:7px;
                height:12px;

                border-right:2px solid #9b674d;
            }}


            .mouth {{
                position:absolute;

                left:28px;
                top:51px;

                width:18px;
                height:7px;

                border-bottom:
                    3px solid #7b3f3f;

                border-radius:50%;
            }}


            /* BODY */

            .body {{
                position:absolute;

                top:70px;
                left:27px;

                width:78px;
                height:92px;

                background:#25364a;

                border-radius:20px 20px 8px 8px;
            }}


            .shirt {{
                position:absolute;

                left:22px;
                top:0;

                width:34px;
                height:90px;

                background:#f2eee7;

                border-radius:5px;
            }}


            /* ARMS */

            .arm {{
                position:absolute;

                width:20px;
                height:75px;

                background:#25364a;

                border-radius:15px;

                transform-origin:top center;
            }}


            .left-arm {{
                left:-13px;
                top:10px;

                animation:
                    leftGesture
                    2.2s
                    ease-in-out
                    infinite;
            }}


            .right-arm {{
                right:-13px;
                top:10px;

                animation:
                    rightGesture
                    2.6s
                    ease-in-out
                    infinite;
            }}


            @keyframes leftGesture {{

                0% {{
                    transform:rotate(15deg);
                }}

                45% {{
                    transform:rotate(-38deg);
                }}

                70% {{
                    transform:rotate(-10deg);
                }}

                100% {{
                    transform:rotate(15deg);
                }}

            }}


            @keyframes rightGesture {{

                0% {{
                    transform:rotate(-20deg);
                }}

                40% {{
                    transform:rotate(38deg);
                }}

                70% {{
                    transform:rotate(10deg);
                }}

                100% {{
                    transform:rotate(-20deg);
                }}

            }}


            .hand {{
                position:absolute;

                bottom:-9px;
                left:-1px;

                width:22px;
                height:22px;

                border-radius:50%;

                background:#d69b72;
            }}


            /* LEGS */

            .legs {{
                position:absolute;

                top:150px;
                left:34px;

                width:65px;
                height:85px;
            }}


            .leg {{
                position:absolute;

                top:0;

                width:23px;
                height:75px;

                background:#252b36;

                border-radius:10px;

                transform-origin:top center;
            }}


            .left-leg {{
                left:5px;

                animation:
                    leftLeg
                    0.9s
                    ease-in-out
                    infinite;
            }}


            .right-leg {{
                right:5px;

                animation:
                    rightLeg
                    0.9s
                    ease-in-out
                    infinite;
            }}


            @keyframes leftLeg {{

                0% {{
                    transform:rotate(12deg);
                }}

                50% {{
                    transform:rotate(-15deg);
                }}

                100% {{
                    transform:rotate(12deg);
                }}

            }}


            @keyframes rightLeg {{

                0% {{
                    transform:rotate(-15deg);
                }}

                50% {{
                    transform:rotate(12deg);
                }}

                100% {{
                    transform:rotate(-15deg);
                }}

            }}


            .shoe {{
                position:absolute;

                bottom:-8px;
                left:-6px;

                width:34px;
                height:15px;

                border-radius:12px;

                background:#f4f4f4;
            }}


            /* PODIUM */

            .podium {{
                position:absolute;

                right:9%;
                bottom:70px;

                width:150px;
                height:110px;

                background:#704b2d;

                border-radius:8px;

                z-index:7;
            }}


            .laptop {{
                position:absolute;

                top:-22px;
                left:28px;

                width:90px;
                height:48px;

                background:#202a35;

                border-radius:5px;

                transform:skewX(-8deg);
            }}


            /* DESKS */

            .desk {{
                position:absolute;

                bottom:0;

                width:190px;
                height:55px;

                background:#704b2d;

                border-radius:8px 8px 0 0;

                opacity:.8;
            }}


            .desk-one {{
                left:-20px;
            }}


            .desk-two {{
                left:35%;
            }}


            .desk-three {{
                right:-20px;
            }}


            /* STATUS */

            .faculty-status {{
                position:absolute;

                bottom:18px;
                left:50%;

                transform:translateX(-50%);

                padding:10px 22px;

                border-radius:30px;

                background:rgba(7,25,45,.9);

                color:#8fe3ff;

                font-size:15px;

                z-index:20;
            }}

            </style>
            """,
            height=650
        )

       
        # FIND LOCAL PDF CONTENT
       

        chunks = st.session_state.chunks

        # Use simple keyword matching.
        # This works even when Gemini is unavailable.

        topic_words = [
            word.lower()
            for word in current_topic
            .replace(":", " ")
            .replace("-", " ")
            .split()
            if len(word) > 2
        ]

        scored_chunks = []

        for chunk in chunks:

            chunk_lower = chunk.lower()

            score = 0

            for word in topic_words:

                if word in chunk_lower:
                    score += 1

            scored_chunks.append(
                (score, chunk)
            )

        scored_chunks.sort(
            key=lambda x: x[0],
            reverse=True
        )

        relevant_chunks = [
            item[1]
            for item in scored_chunks[:3]
        ]

        context = "\n\n".join(
            relevant_chunks
        )

       
        # LOCAL FACULTY EXPLANATION
       

        topic_key =(f"local_lesson_{current_index}_{selected_language}")

        if topic_key not in st.session_state:

            if selected_language == "English":

                lesson = f"""
👨‍🏫 Welcome to today's lesson!

📚 Topic:
{current_topic}

Let's understand this topic from your uploaded PDF.

━━━━━━━━━━━━━━━━━━━━

🔹 What we will study

The following information is available in your PDF:

{context}

━━━━━━━━━━━━━━━━━━━━

🔹 Faculty Explanation

Read the above material carefully.

I have selected the most relevant part of your PDF
for this topic so that we can study it step-by-step.

Focus on:
• Important definitions
• Important formulas
• Numerical values
• Given conditions
• Final results

━━━━━━━━━━━━━━━━━━━━

💡 Study Tip

First understand what information is given in the
question. Then identify the formula or concept
being used. Finally, solve it step-by-step.

💬 You can now ask me a question about this topic.
"""

            elif selected_language == "Hindi":

                lesson = f"""
👨‍🏫 आज की क्लास में आपका स्वागत है!

📚 Topic:
{current_topic}

आइए आपके द्वारा upload की गई PDF के आधार पर
इस topic को समझते हैं।

━━━━━━━━━━━━━━━━━━━━

🔹 PDF में उपलब्ध जानकारी

आपकी PDF से इस topic से संबंधित information:

{context}

━━━━━━━━━━━━━━━━━━━━

🔹 Faculty Explanation

सबसे पहले PDF में दिए गए important points को समझें।

ध्यान दें:
• Important definitions
• Important formulas
• दिए गए numerical values
• Question में दी गई conditions
• Final result

━━━━━━━━━━━━━━━━━━━━

💡 Study Tip

पहले question में दी गई information को समझें।
फिर देखिए कि कौन-सा formula या concept use करना है।
उसके बाद step-by-step solution करें।

💬 अब आप इस topic से related कोई भी question पूछ सकते हैं।
"""

            else:

                lesson = f"""
👨‍🏫 Welcome to today's class!

📚 Topic:
{current_topic}

Chaliye is topic ko aapki uploaded PDF ke basis
par step-by-step samajhte hain.

━━━━━━━━━━━━━━━━━━━━

🔹 PDF se relevant information

Aapki PDF me is topic se related information:

{context}

━━━━━━━━━━━━━━━━━━━━

🔹 Faculty Explanation

Sabse pehle question me di hui information ko
carefully samjho.

Focus karo:

• Important definitions
• Important formulas
• Given numerical values
• Conditions
• Final result

━━━━━━━━━━━━━━━━━━━━

💡 Study Tip

Pehle identify karo ki question me kya given hai.
Phir dekho kaunsa formula ya concept use hoga.
Uske baad step-by-step solve karo.

💬 Ab aap is topic se related cross-question pooch sakte ho.
"""

            st.session_state[topic_key] = lesson

      
        # SHOW LESSON
      

        st.markdown(
            st.session_state[topic_key]
        )

       
        # VOICE
       

        st.markdown("---")

        st.markdown("### 🔊 Listen to Explanation")

        speech_text = (
            st.session_state[topic_key]
            .replace("👨‍🏫", "")
            .replace("📚", "")
            .replace("🔹", "")
            .replace("💡", "")
            .replace("💬", "")
        )

        if selected_language == "Hindi":

            voice_language = "hi-IN"

        elif selected_language == "Hinglish":

            voice_language = "en-IN"

        else:

            voice_language = "en-US"

        import json

        speech_text_js = json.dumps(
            speech_text,
            ensure_ascii=False
        )

        st.components.v1.html(
            f"""
            <div style="
                display:flex;
                gap:12px;
                align-items:center;
                padding:10px;
            ">

                <button
                    onclick="startSpeech()"
                    style="
                        padding:10px 20px;
                        border:none;
                        border-radius:8px;
                        background:#2196F3;
                        color:white;
                        font-size:16px;
                        cursor:pointer;
                    "
                >
                    ▶️ Speak
                </button>

                <button
                    onclick="stopSpeech()"
                    style="
                        padding:10px 20px;
                        border:none;
                        border-radius:8px;
                        background:#E53935;
                        color:white;
                        font-size:16px;
                        cursor:pointer;
                    "
                >
                    ⏹ Stop
                </button>

                <span id="status"
                    style="
                        font-size:15px;
                        color:#cccccc;
                    "
                >
                    Ready
                </span>

            </div>

            <script>

                const text = {speech_text_js};

                const language = "{voice_language}";

                let speech = null;

                function startSpeech() {{

                    window.speechSynthesis.cancel();

                    speech =
                        new SpeechSynthesisUtterance(text);

                    speech.lang = language;

                    speech.rate = 0.9;

                    speech.pitch = 1.0;

                    speech.volume = 1.0;

                    document.getElementById(
                        "status"
                    ).innerText = "🔊 Speaking...";

                    speech.onend = function() {{

                        document.getElementById(
                            "status"
                        ).innerText = "✅ Finished";

                    }};

                    speech.onerror = function() {{

                        document.getElementById(
                            "status"
                        ).innerText = "⚠️ Voice error";

                    }};

                    window.speechSynthesis.speak(
                        speech
                    );
                }}

                function stopSpeech() {{

                    window.speechSynthesis.cancel();

                    document.getElementById(
                        "status"
                    ).innerText = "⏹ Stopped";
                }}

            </script>
            """,
            height=75
        )
        # ---------------------------------------------
        # NEXT TOPIC
        # ---------------------------------------------

        st.markdown("---")

        if st.button(
            "➡️ Next Topic",
            key="next_topic_button"
        ):

            st.session_state.current_topic = (
                current_index + 1
            )

            st.rerun()

        # ---------------------------------------------
        # CROSS QUESTIONING
        # ---------------------------------------------

               # -----------------------------
        # CROSS QUESTIONING
        # -----------------------------

        st.markdown("---")

        st.markdown("### 💬 Cross-Question Your Faculty")

        question = st.text_input(
            "Ask your question:",
            placeholder="Example: Why does impedance occur?",
            key=f"faculty_question_{current_index}"
        )

        if question:

            with st.spinner("👨‍🏫 Faculty is searching your PDF..."):

                # ---------------------------------
                # EMBED STUDENT QUESTION
                # ---------------------------------

                question_result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=question
                )

                question_embedding = np.array(
                    [question_result.embeddings[0].values],
                    dtype="float32"
                )

                # ---------------------------------
                # SEARCH FAISS
                # ---------------------------------

                qk = min(
                    3,
                    len(st.session_state.chunks)
                )

                distances, indices = (
                    st.session_state.faiss_index.search(
                        question_embedding,
                        k=qk
                    )
                )

                relevant_chunks = []

                for idx in indices[0]:

                    if idx != -1:

                        relevant_chunks.append(
                            st.session_state.chunks[idx]
                        )

                # ---------------------------------
                # SHOW FACULTY RESPONSE
                # ---------------------------------

                st.markdown("### 👨‍🏫 Faculty")

                if relevant_chunks:

                    st.write(
                        "Maine aapke question ke liye "
                        "uploaded PDF se ye relevant information find ki:"
                    )

                    for i, chunk in enumerate(
                        relevant_chunks,
                        start=1
                    ):

                        with st.expander(
                            f"📖 Relevant PDF Content {i}"
                        ):

                            st.write(chunk)

                    st.success(
                        "💡 Ye answer directly aapke uploaded "
                        "PDF ke relevant content se retrieve hua hai."
                    )

                else:

                    st.warning(
                        "⚠️ Is question ke liye uploaded PDF "
                        "mein relevant information nahi mili."
                    )



    elif menu == "🎭 Virtual Classroom":

        import os
        import requests
        import time

        st.markdown("## 🎭 Virtual Classroom")

        # -----------------------------
        # CHECK PDF
        # -----------------------------
        if "chunks" not in st.session_state:
            st.warning("📚 Please upload a PDF first from My PDFs.")
            st.stop()

        # -----------------------------
        # GET TOPICS
        # -----------------------------
        topic_lines = []

        if "topics" in st.session_state:
            topic_lines = [
                line.strip()
                for line in st.session_state.topics.split("\n")
                if line.strip()
            ]

        if not topic_lines:
            st.warning(
                "⚠️ No topics available. Please upload your PDF again."
            )
            st.stop()

        # -----------------------------
        # CURRENT TOPIC
        # -----------------------------
        current_index = (
            st.session_state.get("current_topic", 0)
            % len(topic_lines)
        )

        current_topic = topic_lines[current_index]

        # -----------------------------
        # CLASSROOM HEADER
        # -----------------------------
        st.info("🎓 Faculty is ready for today's lecture.")

        st.markdown("### 📚 Today's Topic")

        st.success(current_topic)

        st.markdown("---")

        # -----------------------------
        # DIGITAL BOARD
        # -----------------------------
        st.markdown("### 📋 Digital Board")

        with st.container(border=True):

            st.markdown("## " + current_topic)

            st.write(
                "The faculty will explain this topic using "
                "the uploaded PDF material."
            )

        st.markdown("---")

        # -----------------------------
        # FACULTY VIDEO
        # -----------------------------
        st.markdown("### 👨‍🏫 AI Virtual Faculty")

        st.write(
            f"Currently teaching: **{current_topic}**"
        )

        st.markdown("")

        # -----------------------------
        # START LECTURE
        # -----------------------------
        if st.button(
            "▶️ Start Faculty Lecture",
            use_container_width=True
        ):

            api_key = os.getenv("DID_API_KEY")

            if not api_key:

                st.error(
                    "❌ D-ID API key not found. "
                    "Please restart VS Code after setting the key."
                )

            else:

                # -----------------------------
                # FIND RELEVANT PDF CONTENT
                # -----------------------------

                topic_words = set(
                    current_topic.lower()
                    .replace(":", "")
                    .replace("-", " ")
                    .split()
                )

                relevant_chunks = []

                for chunk in st.session_state.chunks:

                    chunk_lower = chunk.lower()

                    score = sum(
                        1
                        for word in topic_words
                        if len(word) > 2 and word in chunk_lower
                    )

                    if score > 0:
                        relevant_chunks.append(
                            (score, chunk)
                        )

                relevant_chunks.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                if relevant_chunks:

                    lecture_chunks = [
                        chunk
                        for score, chunk
                        in relevant_chunks[:3]
                    ]

                else:

                    lecture_chunks = st.session_state.chunks[:3]

                pdf_content = "\n\n".join(
                    lecture_chunks
                )

               # -----------------------------
                # CREATE LECTURE SCRIPT
                # -----------------------------

               # -----------------------------
# CREATE FACULTY EXPLANATION
# -----------------------------

                selected_language = st.session_state.get(
                    "selected_language",
                    "English"
                )

                faculty_prompt = f"""
                You are a college faculty member.

                Teach this topic using ONLY the provided PDF material.

                Topic:
                {current_topic}

                PDF Material:
                {pdf_content[:2500]}

                Language:
                {selected_language}

                Instructions:
                - Explain like a real teacher.
                - Keep it simple and natural.
                - Do not invent information.
                - Focus on the main concept.
                - Keep the lecture short.
                - Return only the speech script.
                """

                try:

                    # PRIMARY: GEMINI
                    lecture_response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=faculty_prompt
                    )

                    lecture_script = lecture_response.text.strip()

                    lecture_source = "Gemini"

                except Exception:

                    # FALLBACK: LOCAL FACULTY
                    lecture_script = create_local_faculty_explanation(
                        current_topic,
                        pdf_content,
                        selected_language
                    )

                    lecture_source = "Local Fallback"

                lecture_script = lecture_script[:700]

                st.info(
                    f"👨‍🏫 Lecture generated using: **{lecture_source}**"
                )
                st.markdown("### 🎙️ Faculty Speech Preview")
                st.write(lecture_script)





                               
                # TEST FACULTY VOICE
                

                voice_language = {
                    "Hindi": "hi-IN",
                    "Hinglish": "en-IN",
                    "English": "en-US"
                }.get(selected_language, "en-US")

                import streamlit.components.v1 as components

                voice_text = lecture_script.replace(
                    "'", "\\'"
                ).replace("\n", " ")

                components.html(
                    f"""
                    <script>

                        function speakLecture() {{

                            window.speechSynthesis.cancel();

                            const voices = window.speechSynthesis.getVoices();

                            const language = "{voice_language}";

                            let selectedVoice = null;

                            if (language === "hi-IN") {{

                                selectedVoice = voices.find(
                                    voice => voice.name === "Google हिन्दी"
                                );

                            }} else if (language === "en-IN") {{

                                selectedVoice = voices.find(
                                    voice => voice.lang === "en-IN"
                                );

                            }} else {{

                                selectedVoice = voices.find(
                                    voice => voice.lang === "en-US"
                                );

                            }}

                            const speech =
                                new SpeechSynthesisUtterance(
                                    '{voice_text}'
                                );

                            speech.lang = language;

                            if (selectedVoice) {{
                                speech.voice = selectedVoice;
                            }}

                            speech.rate = 0.9;
                            speech.pitch = 1.0;

                            window.speechSynthesis.speak(speech);
                        }}


                        function stopLecture() {{
                            window.speechSynthesis.cancel();
                        }}

                    </script>


                    <button onclick="speakLecture()"
                        style="
                        padding:12px 22px;
                        font-size:16px;
                        border-radius:10px;
                        border:none;
                        cursor:pointer;
                        ">
                        🔊 Play Faculty Voice
                    </button>


                    <button onclick="stopLecture()"
                        style="
                        padding:12px 22px;
                        font-size:16px;
                        border-radius:10px;
                        border:none;
                        cursor:pointer;
                        margin-left:10px;
                        ">
                        ⏹ Stop
                    </button>
                    """,
                    height=70
                )
               # st.stop()
                # -----------------------------
                # D-ID REQUEST
                # -----------------------------
                # -----------------------------
                # D-ID REQUEST
                # -----------------------------

                headers = {
                    "accept": "application/json",
                    "authorization": f"Basic {api_key}",
                    "content-type": "application/json"
                }

                data = {
                    "presenter_id": "v2_public_Amber@0zSz8kflCN",
                    "script": {
                        "type": "text",
                        "input": lecture_script
                    }
                }

                try:

                    with st.spinner(
                        "👩‍🏫 Faculty is preparing your lecture..."
                    ):

                        response = requests.post(
                            "https://api.d-id.com/clips",
                            headers=headers,
                            json=data,
                            timeout=60
                        )

                    if response.status_code not in [200, 201]:

                        st.error(
                            "❌ D-ID video generation failed."
                        )

                        st.code(
                            response.text
                        )

                    else:

                        result = response.json()

                        video_id = result.get("id")

                        st.info(
                            "🎬 Faculty video is being generated..."
                        )

                        status_url = (
                            f"https://api.d-id.com/clips/{video_id}"
                        )

                        video_url = None

                        # -----------------------------
                        # CHECK VIDEO STATUS
                        # -----------------------------

                        status_placeholder = st.empty()

                        for attempt in range(18):

                            time.sleep(5)

                            try:

                                status_response = requests.get(
                                    status_url,
                                    headers={
                                        "accept": "application/json",
                                        "authorization": f"Basic {api_key}"
                                    },
                                    timeout=30
                                )

                                status_response.raise_for_status()

                                status_data = status_response.json()

                                status = status_data.get("status")

                                status_placeholder.info(
                                    f"🎬 Faculty video status: **{status}**"
                                )

                                if status == "done":

                                    video_url = status_data.get("result_url")

                                    break

                                if status == "error":

                                    st.error(
                                        "❌ D-ID video generation failed."
                                    )

                                    st.code(str(status_data))

                                    break

                            except requests.exceptions.Timeout:

                                status_placeholder.warning(
                                    "⚠️ Status check timed out. Retrying..."
                                )

                            except requests.exceptions.RequestException as e:

                                status_placeholder.warning(
                                    "⚠️ Network error. Retrying..."
                                )
                        # -----------------------------
                        # SHOW VIDEO
                        # -----------------------------

                        if video_url:

                            st.success(
                                "🎉 Faculty lecture is ready!"
                            )

                            st.video(
                                video_url
                            )

                            st.session_state[
                                "virtual_classroom_video"
                            ] = video_url

                        else:

                            st.warning(
                                "⚠️ Faculty video is still processing. "
                                "Please try again in a moment."
                            )

                except requests.exceptions.Timeout:

                    st.error(
                        "❌ D-ID connection timed out. "
                        "Please try the lecture again."
                    )

                except requests.exceptions.RequestException as e:

                    st.error(
                        "❌ Network error while connecting to D-ID."
                    )

                    st.code(
                        str(e)
                    )

        # -----------------------------
        # NAVIGATION
        # -----------------------------

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            if st.button("⬅️ Previous Topic"):

                if current_index > 0:

                    st.session_state.current_topic = (
                        current_index - 1
                    )

                    st.rerun()

        with col2:

            st.write(
                f"Topic {current_index + 1} / "
                f"{len(topic_lines)}"
            )

        with col3:

            if st.button("Next Topic ➡️"):

                if current_index < len(topic_lines) - 1:

                    st.session_state.current_topic = (
                        current_index + 1
                    )

                    st.rerun()
    # =================================================
    # QUIZ - EXACTLY 5 QUESTIONS FOR ONE TOPIC
    # =================================================

       # =================================================
    # QUIZ - GEMINI + LOCAL FALLBACK
    # =================================================

    elif menu == "📝 Quiz":

        if "faiss_index" not in st.session_state or "chunks" not in st.session_state:
            st.warning(
                "📚 Please upload a PDF first from **My PDFs**."
            )
            st.stop()

        if "topics" not in st.session_state:
            st.warning(
                "⚠️ Please upload your PDF from **My PDFs** "
                "first so topics can be detected."
            )
            st.stop()

        st.markdown("## 📝 Topic-wise Quiz")

        st.write(
            "Select a topic and test your understanding "
            "using your uploaded PDF."
        )

        # ---------------------------------------------
        # GET TOPICS
        # ---------------------------------------------

        topic_lines = [
            line.strip()
            for line in st.session_state.topics.split("\n")
            if line.strip()
        ]

        if not topic_lines:
            st.warning(
                "⚠️ No topics are available."
            )
            st.stop()

        selected_topic = st.selectbox(
            "📚 Select a topic",
            topic_lines,
            key="selected_quiz_topic"
        )

        # ---------------------------------------------
        # RESET QUIZ WHEN TOPIC CHANGES
        # ---------------------------------------------

        if (
            st.session_state.get("quiz_topic")
            != selected_topic
        ):

            st.session_state.pop(
                "quiz_questions",
                None
            )

            st.session_state.pop(
                "quiz_submitted",
                None
            )

        # ---------------------------------------------
        # GENERATE 5 QUESTIONS
        # ---------------------------------------------

        if st.button(
            "🎯 Generate 5 Questions",
            key="generate_quiz_button"
        ):

            with st.spinner(
                "🧠 Faculty is preparing your quiz..."
            ):

                # -------------------------------------
                # STEP 1: FIND RELEVANT PDF CONTENT
                # -------------------------------------

                try:

                    topic_result = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=selected_topic
                    )

                    topic_embedding = np.array(
                        [topic_result.embeddings[0].values],
                        dtype="float32"
                    )

                    qk = min(
                        5,
                        len(st.session_state.chunks)
                    )

                    distances, indices = (
                        st.session_state.faiss_index.search(
                            topic_embedding,
                            k=qk
                        )
                    )

                    relevant_chunks = []

                    for idx in indices[0]:

                        if idx != -1:

                            relevant_chunks.append(
                                st.session_state.chunks[idx]
                            )

                    quiz_context = "\n\n".join(
                        relevant_chunks
                    )

                except Exception as e:

                    st.error(
                        "⚠️ PDF content retrieve nahi ho paya."
                    )

                    st.stop()

                # -------------------------------------
                # STEP 2: GEMINI QUIZ GENERATION
                # -------------------------------------

                quiz_prompt = f"""
You are EduAvatar, an AI college faculty member.

Create a quiz ONLY from the supplied PDF context.

TOPIC:
{selected_topic}

PDF CONTEXT:
{quiz_context}

Create EXACTLY 5 multiple-choice questions.

Return ONLY valid JSON.

Use exactly this format:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": 0,
    "explanation": "Short explanation"
  }}
]

RULES:

1. Exactly 5 questions.
2. Exactly 4 options per question.
3. answer must be 0, 1, 2, or 3.
4. Only ONE option should be correct.
5. Questions must be based ONLY on the supplied PDF context.
6. Do NOT use outside knowledge.
7. Do NOT repeat questions.
8. Questions should be suitable for a college student.
9. Make the questions concept-based.
10. Vary the position of the correct answer.
"""

                gemini_success = False

                # -------------------------------------
                # TRY GEMINI
                # -------------------------------------

                try:

                    quiz_response = (
                        client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=quiz_prompt
                        )
                    )

                    raw_quiz = (
                        quiz_response.text
                        .strip()
                    )

                    # Remove markdown code fences
                    if raw_quiz.startswith("```"):

                        raw_quiz = (
                            raw_quiz
                            .replace(
                                "```json",
                                "",
                                1
                            )
                            .replace(
                                "```",
                                ""
                            )
                            .strip()
                        )

                    questions = json.loads(
                        raw_quiz
                    )

                    # ---------------------------------
                    # VALIDATE GEMINI RESPONSE
                    # ---------------------------------

                    if (
                        not isinstance(
                            questions,
                            list
                        )
                        or len(questions) != 5
                    ):
                        raise ValueError(
                            "Gemini did not return "
                            "exactly 5 questions."
                        )

                    for q in questions:

                        if not isinstance(
                            q,
                            dict
                        ):
                            raise ValueError(
                                "Invalid question format."
                            )

                        if not isinstance(
                            q.get("question"),
                            str
                        ):
                            raise ValueError(
                                "Invalid question."
                            )

                        if not isinstance(
                            q.get("options"),
                            list
                        ):
                            raise ValueError(
                                "Invalid options."
                            )

                        if len(
                            q["options"]
                        ) != 4:
                            raise ValueError(
                                "Each question must "
                                "have 4 options."
                            )

                        if not isinstance(
                            q.get("answer"),
                            int
                        ):
                            raise ValueError(
                                "Invalid answer."
                            )

                        if q["answer"] not in range(4):
                            raise ValueError(
                                "Answer must be "
                                "between 0 and 3."
                            )

                    # Gemini worked successfully

                    st.session_state.quiz_questions = (
                        questions
                    )

                    st.session_state.quiz_topic = (
                        selected_topic
                    )

                    st.session_state.quiz_submitted = (
                        False
                    )

                    st.session_state.quiz_source = (
                        "Gemini"
                    )

                    gemini_success = True

                    st.success(
                        "🤖 Gemini Faculty generated "
                        "5 questions from your PDF!"
                    )

                # -------------------------------------
                # GEMINI FAILED → LOCAL FALLBACK
                # -------------------------------------

                except Exception as e:

                    st.warning(
                        "⚠️ Gemini generation is "
                        "temporarily unavailable."
                    )

                    st.info(
                        "🔄 EduAvatar is switching to "
                        "Local Quiz Mode so you can "
                        "continue studying."
                    )

                    # ---------------------------------
                    # LOCAL FALLBACK QUIZ
                    # ---------------------------------

                    questions = []

                    for chunk in relevant_chunks:

                        if len(questions) >= 5:
                            break

                        # Clean PDF text

                        clean_text = (
                            chunk
                            .replace("\n", " ")
                            .replace("  ", " ")
                            .strip()
                        )

                        # Split into sentences

                        sentences = [
                            s.strip()
                            for s in clean_text.split(".")
                            if len(
                                s.strip()
                            ) > 30
                        ]

                        for sentence in sentences:

                            if len(questions) >= 5:
                                break

                            words = (
                                sentence.split()
                            )

                            if len(words) < 6:
                                continue

                            correct_answer = (
                                sentence[:180]
                            )

                            # Generic distractors
                            options = [
                                correct_answer,
                                "This information is not given in the PDF.",
                                "This statement is unrelated to the selected topic.",
                                "The PDF provides the opposite information."
                            ]

                            # Rotate correct answer
                            # position so it is not
                            # always option 1.

                            position = (
                                len(questions) % 4
                            )

                            rotated_options = (
                                options[:]
                            )

                            correct_value = (
                                rotated_options.pop(0)
                            )

                            rotated_options.insert(
                                position,
                                correct_value
                            )

                            questions.append(
                                {
                                    "question":
                                        (
                                            "According to "
                                            "the uploaded PDF, "
                                            "which statement "
                                            "is correct?"
                                        ),

                                    "options":
                                        rotated_options,

                                    "answer":
                                        position,

                                    "explanation":
                                        (
                                            "This answer was "
                                            "retrieved from "
                                            "the relevant "
                                            "content of your "
                                            "uploaded PDF."
                                        )
                                }
                            )

                    # ---------------------------------
                    # SAVE FALLBACK QUESTIONS
                    # ---------------------------------

                    if len(questions) >= 5:

                        st.session_state.quiz_questions = (
                            questions[:5]
                        )

                        st.session_state.quiz_topic = (
                            selected_topic
                        )

                        st.session_state.quiz_submitted = (
                            False
                        )

                        st.session_state.quiz_source = (
                            "Local PDF"
                        )

                        st.success(
                            "📚 5 questions are ready "
                            "from your PDF!"
                        )

                    else:

                        st.error(
                            "⚠️ This topic does not have "
                            "enough suitable PDF content "
                            "to create 5 questions."
                        )

        # ---------------------------------------------
        # DISPLAY QUIZ
        # ---------------------------------------------

        if (
            st.session_state.get(
                "quiz_questions"
            )
            and
            st.session_state.get(
                "quiz_topic"
            )
            == selected_topic
        ):

            st.markdown("---")

            # Show source

            quiz_source = (
                st.session_state.get(
                    "quiz_source",
                    "Unknown"
                )
            )

            if quiz_source == "Gemini":

                st.success(
                    "🤖 Quiz generated by Gemini Faculty"
                )

            else:

                st.info(
                    "📚 Quiz generated from your "
                    "uploaded PDF using Local Mode"
                )

            st.markdown(
                f"### 📖 {selected_topic}"
            )

            answers = []

          
            # QUESTIONS
          

            for i, q in enumerate(
                st.session_state.quiz_questions
            ):

                st.markdown(
                    f"#### Q{i + 1}. "
                    f"{q['question']}"
                )

                answer = st.radio(
                    "Choose your answer:",
                    q["options"],
                    key=f"quiz_answer_{i}"
                )

                answers.append(
                    q["options"].index(
                        answer
                    )
                )

        
            # SUBMIT QUIZ
          

            if st.button(
                "✅ Submit Quiz",
                key="submit_quiz_button"
            ):

                score = 0

                for i, q in enumerate(
                    st.session_state.quiz_questions
                ):

                    if (
                        answers[i]
                        == q["answer"]
                    ):

                        score += 1

                st.session_state.quiz_score = (
                    score
                )

                st.session_state.quiz_submitted = (
                    True
                )

               
                # SAVE QUIZ HISTORY
               

                history = st.session_state.get(
                    "quiz_history",
                    []
                )

                history.append(
                    {
                        "topic":
                            selected_topic,

                        "score":
                            score,

                        "total":
                            5
                    }
                )

                st.session_state.quiz_history = (
                    history
                )

          
            # SHOW RESULT
          

            if st.session_state.get(
                "quiz_submitted"
            ):

                score = (
                    st.session_state.quiz_score
                )

                percentage = (
                    score / 5
                ) * 100

                st.markdown("---")

                st.success(
                    f"🎉 Your Score: {score}/5"
                )

                st.progress(
                    percentage / 100
                )

                st.write(
                    f"📊 **Percentage: "
                    f"{percentage:.0f}%**"
                )

                if score == 5:

                    st.info(
                        "🌟 Excellent! "
                        "All 5 answers are correct."
                    )

                elif score >= 3:

                    st.info(
                        "👍 Good job! "
                        "Review the concepts "
                        "you missed once more."
                    )

                else:

                    st.info(
                        "📚 Revise this topic "
                        "and try the quiz again."
                    )

               
                # ANSWER REVIEW
                

                st.markdown(
                    "### 📌 Answer Review"
                )

                for i, q in enumerate(
                    st.session_state.quiz_questions
                ):

                    correct_option = (
                        q["options"][
                            q["answer"]
                        ]
                    )

                    st.write(
                        f"**Q{i + 1} — "
                        f"Correct Answer:** "
                        f"{correct_option}"
                    )

                    st.caption(
                        q.get(
                            "explanation",
                            "Review the relevant "
                            "concept in your PDF."
                        )
                    )
  
# ANALYTICS + MACHINE LEARNING


    elif menu == "📊 Analytics":

        st.markdown("## 📊 Learning Analytics")

        history = st.session_state.get("quiz_history", [])

        if not history:

            st.info(
                "📝 Attempt a topic quiz to generate your learning analytics."
            )

        else:

            
            # BASIC CALCULATIONS
        

            total_quizzes = len(history)

            total_score = sum(
                item["score"]
                for item in history
            )

            total_questions = sum(
                item["total"]
                for item in history
            )

            overall_percentage = (
                total_score / total_questions
            ) * 100


        
            # TOP CARDS
           

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📝 Quizzes Attempted",
                    total_quizzes
                )

            with col2:
                st.metric(
                    "❓ Questions Attempted",
                    total_questions
                )

            with col3:
                st.metric(
                    "🎯 Overall Score",
                    f"{overall_percentage:.0f}%"
                )


            
            # PREPARE TOPIC DATA
           

            topic_data = {}

            for item in history:

                topic = item["topic"]
                score = item["score"]
                total = item["total"]

                if topic not in topic_data:

                    topic_data[topic] = {
                        "score": 0,
                        "total": 0
                    }

                topic_data[topic]["score"] += score
                topic_data[topic]["total"] += total


          
            # TOPIC PERFORMANCE GRAPH
           

            st.markdown("---")

            st.markdown(
                "### 📚 Topic-wise Performance"
            )

            graph_data = []

            for topic, data in topic_data.items():

                percentage = (
                    data["score"] /
                    data["total"]
                ) * 100

                graph_data.append({
                    "Topic": topic,
                    "Score": round(percentage, 1)
                })


            df_topic = pd.DataFrame(graph_data)

            df_topic = df_topic.set_index("Topic")


            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4.5))

            bars = ax.bar(
                df_topic.index,
                df_topic["Score"],
                width=0.6
            )

            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")
            ax.set_xlabel("Topics")
            ax.set_title("Topic-wise Performance")

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.25
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar in bars:
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f"{height:.0f}%",
                    ha="center",
                    fontsize=10
                )

            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


          
            # SCORE TABLE
        

            st.markdown(
                "### 📋 Detailed Performance"
            )

            for topic, data in topic_data.items():

                percentage = (
                    data["score"] /
                    data["total"]
                ) * 100

                if percentage >= 60:

                    st.success(
                        f"✅ **{topic}** — "
                        f"{data['score']}/{data['total']} "
                        f"({percentage:.0f}%)"
                    )

                else:

                    st.warning(
                        f"⚠️ **{topic}** — "
                        f"{data['score']}/{data['total']} "
                        f"({percentage:.0f}%)"
                    )


          
            # QUIZ PROGRESS GRAPH
           

            st.markdown("---")

            st.markdown(
                "### 📈 Quiz Performance Progress"
            )

            progress_data = []

            for i, item in enumerate(history):

                percentage = (
                    item["score"] /
                    item["total"]
                ) * 100

                progress_data.append({
                    "Quiz": f"Quiz {i + 1}",
                    "Score": round(percentage, 1)
                })


            df_progress = pd.DataFrame(
                progress_data
            )

            df_progress = df_progress.set_index(
                "Quiz"
            )


            fig, ax = plt.subplots(figsize=(10, 4.5))

            ax.plot(
                df_progress.index,
                df_progress["Score"],
                marker="o",
                linewidth=2.5
            )

            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")
            ax.set_xlabel("Quiz")
            ax.set_title("Quiz Performance Progress")

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.25
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for x, y in zip(
                df_progress.index,
                df_progress["Score"]
            ):
                ax.text(
                    x,
                    y + 3,
                    f"{y:.0f}%",
                    ha="center",
                    fontsize=10
                )

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

           
            # RANDOM FOREST ML
            

            st.markdown("---")

            st.markdown(
                "### 🤖 AI Learning Analysis"
            )

            st.write(
                "Random Forest is used as a simple "
                "adaptive-learning classifier to identify "
                "topics that may need revision."
            )


            # Training data

            X = []
            y = []

            for score in range(6):

                percentage_value = (
                    score / 5
                ) * 100

                X.append([
                    score,
                    5,
                    percentage_value
                ])

                if score >= 3:
                    y.append(1)

                else:
                    y.append(0)


            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X, y)


          
            # ML PREDICTION
           

            st.markdown(
                "#### 🔍 Topic Prediction"
            )

            weak_topics = []

            for topic, data in topic_data.items():

                percentage = (
                    data["score"] /
                    data["total"]
                ) * 100

                prediction = model.predict([
                    [
                        data["score"],
                        data["total"],
                        percentage
                    ]
                ])[0]


                if prediction == 1:

                    st.success(
                        f"🟢 **{topic}** — "
                        f"Good Understanding ({percentage:.0f}%)"
                    )

                else:

                    st.warning(
                        f"🔴 **{topic}** — "
                        f"Needs Revision ({percentage:.0f}%)"
                    )

                    weak_topics.append(topic)


          
            # RECOMMENDATION
            st.markdown("---")

            st.markdown(
                "### 🎓 Personalized Learning Recommendation"
            )

            if weak_topics:

                st.warning(
                    "The following topics should be revised:"
                )

                for topic in weak_topics:

                    st.write(
                        f"📖 {topic}"
                    )

                st.info(
                    "👨‍🏫 Go to AI Faculty and revise "
                    "these topics before attempting "
                    "the quiz again."
                )

            else:

                st.success(
                    "🌟 No topic is currently marked "
                    "for revision."
                )
    
   

     
    # SETTINGS
   
    elif menu == "⚙️ Settings":

        st.markdown("## ⚙️ Settings")

        if "selected_language" not in st.session_state:
            st.session_state.selected_language = "English"

        language = st.selectbox(
            "Preferred Explanation Language",
            [
                "English",
                "Hindi",
                "Hinglish"
            ],
            index=[
                "English",
                "Hindi",
                "Hinglish"
            ].index(st.session_state.selected_language)
        )

        st.session_state.selected_language = language

        st.success(
            f"🌐 Current Faculty Language: **{language}**"
        )
