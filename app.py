import os
import streamlit as st
import requests
from datetime import datetime
import json
import hashlib
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


USERS_FILE = "users.json"
HISTORY_FILE = "chat_history.json"
MAX_USERS = 5

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def load_chat_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_chat_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def get_user_history(username):
    history = load_chat_history()
    return history.get(username, [])

def save_user_history(username, messages):
    history = load_chat_history()
    history[username] = messages
    save_chat_history(history)

def delete_user_history(username):
    history = load_chat_history()
    if username in history:
        del history[username]
        save_chat_history(history)

def extract_keywords(text):
    symptom_keywords = ['pain', 'fever', 'cough', 'headache', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'weakness', 'dizzy', 'bleeding', 'swelling', 'rash', 'itch', 'burn', 'cold', 'flu', 'sore', 'ache', 'cramp', 'infection']
    medicine_keywords = ['tablet', 'capsule', 'syrup', 'injection', 'mg', 'ml', 'dose', 'antibiotic', 'paracetamol', 'ibuprofen', 'aspirin', 'medication', 'drug', 'prescription']
    
    words = text.lower().split()
    found = []
    for word in words:
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in symptom_keywords or clean_word in medicine_keywords:
            found.append(clean_word)
    return list(set(found))

def generate_medical_receipt_pdf(username, messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Header
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1f77b4'), spaceAfter=12, alignment=1)
    story.append(Paragraph("MEDICAL CONSULTATION RECEIPT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Info
    now = datetime.now()
    info_data = [
        ['Patient Name:', username],
        ['Date:', now.strftime('%Y-%m-%d')],
        ['Time:', now.strftime('%H:%M:%S')]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Extract keywords
    symptom_keywords = set()
    remedy_keywords = set()
    medicine_keywords = set()
    
    for i, msg in enumerate(messages):
        if msg['role'] == 'user' and i > 0:
            keywords = extract_keywords(msg['content'])
            symptom_keywords.update(keywords)
        elif msg['role'] == 'assistant':
            content = msg['content'].lower()
            # Extract medicine names and keywords
            med_words = ['tablet', 'capsule', 'syrup', 'mg', 'ml', 'antibiotic', 'paracetamol', 'ibuprofen', 'aspirin', 'medication', 'drug']
            remedy_words = ['rest', 'drink', 'water', 'exercise', 'diet', 'avoid', 'sleep', 'treatment', 'therapy']
            
            for word in content.split():
                clean = re.sub(r'[^a-z]', '', word)
                if clean in med_words:
                    medicine_keywords.add(clean)
                elif clean in remedy_words:
                    remedy_keywords.add(clean)
    
    # Symptoms Section
    story.append(Paragraph("<b>SYMPTOMS:</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    if symptom_keywords:
        story.append(Paragraph(", ".join(sorted(symptom_keywords)).title(), styles['Normal']))
    else:
        story.append(Paragraph("None recorded", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Remedies Section
    story.append(Paragraph("<b>REMEDIES:</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    if remedy_keywords:
        story.append(Paragraph(", ".join(sorted(remedy_keywords)).title(), styles['Normal']))
    else:
        story.append(Paragraph("None provided", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Medicines Section
    story.append(Paragraph("<b>MEDICINES:</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    if medicine_keywords:
        story.append(Paragraph(", ".join(sorted(medicine_keywords)).title(), styles['Normal']))
    else:
        story.append(Paragraph("None mentioned", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Warning Footer
    warning_style = ParagraphStyle('Warning', parent=styles['Normal'], fontSize=10, textColor=colors.red, alignment=1, spaceAfter=6)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>⚠️ IMPORTANT DISCLAIMER ⚠️</b>", warning_style))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, alignment=1, textColor=colors.HexColor('#666666'))
    story.append(Paragraph("This is an AI-generated consultation summary and NOT a medical prescription.", disclaimer_style))
    story.append(Paragraph("Please consult a certified doctor before taking any medication or treatment.", disclaimer_style))
    story.append(Paragraph("This document is for informational purposes only.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    # Admin account check
    if username == "admin" and password == "admin":
        return True
    
    users = load_users()
    if username in users:
        if users[username]['password'] == hash_password(password):
            users[username]['usage_count'] = users[username].get('usage_count', 0) + 1
            save_users(users)
            return True
    return False

def is_admin(username):
    return username == "admin"

def register_user(username, password):
    users = load_users()
    
    if username in users:
        return False, "User already exists"
    
    if len(users) >= MAX_USERS:
        # Find least used user
        least_used = min(users.keys(), key=lambda x: users[x].get('usage_count', 0))
        del users[least_used]
        st.warning(f"User limit reached. Replaced least used account: {least_used}")
    
    users[username] = {
        'password': hash_password(password),
        'usage_count': 0,
        'created': datetime.now().isoformat()
    }
    save_users(users)
    return True, "User registered successfully"

def admin_panel():
    st.title("🔧 Admin Panel")
    
    tab1, tab2 = st.tabs(["User Management", "Create User"])
    
    with tab1:
        st.subheader("User Management")
        users = load_users()
        
        if users:
            for username, data in users.items():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{username}**")
                with col2:
                    st.write(f"Usage: {data.get('usage_count', 0)}")
                with col3:
                    if st.button("Delete", key=f"del_{username}"):
                        del users[username]
                        save_users(users)
                        delete_user_history(username)
                        st.success(f"User {username} deleted!")
                        st.rerun()
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("Create New User")
        new_username = st.text_input("Username", key="admin_create_user")
        new_password = st.text_input("Password", type="password", key="admin_create_pass")
        
        if st.button("Create User"):
            if new_username and new_password:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill all fields")

def login_page():
    st.title("🔐 MediBot Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="reg_user")
        new_password = st.text_input("Password", type="password", key="reg_pass")
        
        if st.button("Register"):
            if new_username and new_password:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill all fields")


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def get_medical_news():
    try:
        url = "https://newsapi.org/v2/everything?q=medical+health&sortBy=publishedAt&language=en&pageSize=5"
        # Free tier - replace with your NewsAPI key if needed
        headers = {"X-API-Key": "demo_key"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            news_text = f"📰 **Latest Medical News ({datetime.now().strftime('%Y-%m-%d')}):**\n\n"
            
            for i, article in enumerate(data.get('articles', [])[:3], 1):
                title = article.get('title', 'No title')
                description = article.get('description', 'No description')
                published = article.get('publishedAt', '')[:10]
                news_text += f"**{i}. {title}**\n{description}\n*Published: {published}*\n\n"
            
            return news_text
        else:
            return "📰 **Medical News Update:**\n\nHere are some recent medical developments:\n\n1. **COVID-19 Vaccine Updates**: New booster recommendations from health authorities\n2. **Heart Disease Research**: Latest findings on prevention strategies\n3. **Mental Health Awareness**: New treatment approaches being studied\n\n*Note: For the most current news, please visit reputable medical news sources.*"
    except:
        return "📰 **Medical News Update:**\n\nHere are some recent medical developments:\n\n1. **COVID-19 Vaccine Updates**: New booster recommendations from health authorities\n2. **Heart Disease Research**: Latest findings on prevention strategies\n3. **Mental Health Awareness**: New treatment approaches being studied\n\n*Note: For the most current news, please visit reputable medical news sources.*"


def process_user_input(prompt):
    # Handle greetings
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
    if any(greeting in prompt.lower() for greeting in greetings):
        greeting_response = "Hello! I'm MediBot, your medical assistant. I can help you with medical questions, diseases, treatments, and medications. How can I assist you today?"
        st.chat_message('assistant').markdown(greeting_response)
        st.session_state.messages.append({'role':'assistant', 'content': greeting_response})
        save_user_history(st.session_state.username, st.session_state.messages)
        return

    # Handle news requests
    news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'update']
    if any(keyword in prompt.lower() for keyword in news_keywords) and ('medical' in prompt.lower() or 'health' in prompt.lower()):
        news_response = get_medical_news()
        st.chat_message('assistant').markdown(news_response)
        st.session_state.messages.append({'role':'assistant', 'content': news_response})
        save_user_history(st.session_state.username, st.session_state.messages)
        return

    CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """
    
    try: 
        vectorstore=get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response=qa_chain.invoke({'query':prompt})
        result=response["result"]
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role':'assistant', 'content': result})
        save_user_history(st.session_state.username, st.session_state.messages)

    except Exception as e:
        st.error(f"Error: {str(e)}")
def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Show login page if not authenticated
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Admin panel for admin user
    if is_admin(st.session_state.username):
        admin_panel()
        
        # Admin logout
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        return
    
    # Main app for regular users
    st.title(f"🩺 MediBot - Welcome {st.session_state.username}")
    
    # Sidebar with logout and history options
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            if 'messages' in st.session_state:
                del st.session_state.messages
            st.rerun()
        
        st.divider()
        st.subheader("📜 Chat History")
        
        if st.button("🗑️ Delete History"):
            delete_user_history(st.session_state.username)
            st.session_state.messages = []
            st.success("History deleted!")
            st.rerun()
        
        st.divider()
        st.subheader("📄 Medical Receipt")
        
        if st.button("📥 Generate PDF Receipt"):
            if st.session_state.messages:
                pdf_buffer = generate_medical_receipt_pdf(st.session_state.username, st.session_state.messages)
                st.download_button(
                    label="⬇️ Download Receipt",
                    data=pdf_buffer,
                    file_name=f"medical_receipt_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("No chat history to generate receipt!")

    # Load user's chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = get_user_history(st.session_state.username)

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Text input only
    prompt = st.chat_input("Type your medical question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        process_user_input(prompt)
        # Save updated history
        save_user_history(st.session_state.username, st.session_state.messages)

if __name__ == "__main__":
    main()