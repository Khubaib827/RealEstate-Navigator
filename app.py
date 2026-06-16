import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import faiss
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INLINE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Real Estate System",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; color: #1E3A8A; }
    div[data-testid="stMetricLabel"] { font-size: 14px; font-weight: 600; color: #4B5563; }
    .chat-bubble { padding: 12px 16px; border-radius: 12px; margin-bottom: 10px; max-width: 80%; line-height: 1.5; }
    .bot-bubble { background-color: #F3F4F6; color: #1F2937; margin-right: auto; text-align: left; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DEFINING GLOBAL FILE VARIABLES (Perfectly Aligned)
# -----------------------------------------------------------------------------
HOUSING_CSV = "Multan_Housing_Dataset_2000.csv"
REAL_ESTATE_LLM_QA_CSV = "real_estate_llm_qa.csv"
MODEL_PKL = "house_price_model.pkl"

@st.cache_data
def load_housing_data():
    if os.path.exists(HOUSING_CSV):
        return pd.read_csv(HOUSING_CSV)
    return None

df = load_housing_data()
if df is None:
    st.error(f"🚨 Missing Data File: Please place '{HOUSING_CSV}' in this folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PKL)
except:
    model = None

# -----------------------------------------------------------------------------
# 3. INTERNET-FREE OFFLINE VECTOR STORE & RAG ENGINE
# -----------------------------------------------------------------------------
class OfflineRAGEngine:
    def __init__(self, csv_path=REAL_ESTATE_LLM_QA_CSV):
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b') 
        self.index = None
        self.qa_df = None
        self.refresh_pipeline()

    def refresh_pipeline(self):
        if not os.path.exists(self.csv_path):
            self.qa_df = pd.DataFrame(columns=["Category", "Question", "Answer"])
            self.qa_df.to_csv(self.csv_path, index=False)
        else:
            self.qa_df = pd.read_csv(self.csv_path).dropna(subset=["Question", "Answer"])

        if not self.qa_df.empty:
            questions = self.qa_df["Question"].astype(str).tolist()
            tfidf_matrix = self.vectorizer.fit_transform(questions).toarray().astype("float32")
            
            dimension = tfidf_matrix.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(tfidf_matrix)
        else:
            self.index = None

    def ask(self, query):
        if self.index is None or self.qa_df.empty:
            return "Hamare pass filhal koi maloomat save nahi hain."
        
        try:
            query_vector = self.vectorizer.transform([query]).toarray().astype("float32")
            distances, indices = self.index.search(query_vector, 1)
            
            if indices[0][0] != -1:
                idx = indices[0][0]
                if distances[0][0] < 1.9:
                    return self.qa_df.iloc[idx]["Answer"]
        except:
            pass
                
        return "Apka sawal bohot dilchasp hai! Is baray me hamare database me exact jawab nahi mila. Baraye meharbani niche maujood WhatsApp button par click karke hmare agent se rabta karein."

    def add_record(self, category, question, answer):
        new_row = pd.DataFrame([{"Category": category, "Question": question, "Answer": answer}])
        new_row.to_csv(self.csv_path, mode='a', header=False, index=False)
        self.refresh_pipeline()

if "rag" not in st.session_state:
    st.session_state.rag = OfflineRAGEngine()

# -----------------------------------------------------------------------------
# 4. SIDEBAR SELECTION SYSTEM
# -----------------------------------------------------------------------------
st.sidebar.title("🏠 Navigation Menu")
page = st.sidebar.selectbox(
    "Go To Module:",
    ["AI Training Hub (LLM)", "Dashboard", "Price Prediction", "Investment Advisor", "Property Comparison", "Analytics", "Dataset Explorer", "About Project"]
)

# -----------------------------------------------------------------------------
# 5. CORE MODULES IMPLEMENTATION
# -----------------------------------------------------------------------------

# --- PAGE 1: AI TRAINING HUB ---
if page == "AI Training Hub (LLM)":
    st.title("🤖 AI & LLM Training Hub")
    st.markdown("### Contextual Knowledge Management (100% Offline Mode)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("💬 Test Live Semantic RAG Engine")
        user_query = st.text_input("Apna Sawal Likhein (Roman English):", placeholder="e.g., DHA me ghar lene ka tareeqa kya hai?")
        if st.button("Ask AI Bot"):
            if user_query.strip():
                reply = st.session_state.rag.ask(user_query)
                st.markdown(f"<div class='chat-bubble bot-bubble'><b>AI Response:</b><br>{reply}</div>", unsafe_allow_html=True)
            else:
                st.warning("Kuch text type karein.")

    with col2:
        st.subheader("⚙️ Admin Panel: Manage Data")
        
        tab1, tab2 = st.tabs(["➕ Add Record", "🗑️ Delete Record"])
        
        with tab1:
            with st.form("insert_form", clear_on_submit=True):
                cat = st.selectbox("Category:", ["General Inquiry", "Location & Amenities", "Pricing & Budgeting", "Investment & ROI", "Negotiation & Token Money"])
                q = st.text_input("Question Prompt:")
                a = st.text_area("Answer Response Content:")
                if st.form_submit_button("Train Vector Model"):
                    if q.strip() and a.strip():
                        st.session_state.rag.add_record(cat, q.strip(), a.strip())
                        st.success("New Q&A embedded offline inside FAISS indices successfully!")
                        st.rerun()
                    else:
                        st.error("Fields cannot be empty.")
                        
        with tab2:
            if not st.session_state.rag.qa_df.empty:
                all_questions = st.session_state.rag.qa_df["Question"].tolist()
                selected_q_to_delete = st.selectbox("Select Question to Remove:", all_questions)
                
                if st.button("❌ Remove From Database", type="primary", use_container_width=True):
                    # Using the exact matching global variable name here to prevent NameErrors
                    updated_df = st.session_state.rag.qa_df[st.session_state.rag.qa_df["Question"] != selected_q_to_delete]
                    updated_df.to_csv(REAL_ESTATE_LLM_QA_CSV, index=False)
                    st.session_state.rag.refresh_pipeline()
                    st.success("Record deleted successfully!")
                    st.rerun()
            else:
                st.info("Database is empty. Nothing to delete.")

    st.subheader("📋 Present Storage Matrix Index Records")
    st.dataframe(st.session_state.rag.qa_df, use_container_width=True)

# --- PAGE 2: DASHBOARD ---
elif page == "Dashboard":
    st.title("📊 Real Estate System Executive Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sample Properties", f"{len(df):,}")
    c2.metric("Average Asset Price", f"PKR {df['House_Price_PKR'].mean():,.0f}")
    c3.metric("Maximum Market Price Found", f"PKR {df['House_Price_PKR'].max():,.0f}")
    st.subheader("Recent Listings Record Snapshots")
    st.dataframe(df.head(10), use_container_width=True)

# --- PAGE 3: PRICE PREDICTION ---
elif page == "Price Prediction":
    st.title("💰 AI Pricing Valuation Engine")
    if model is None:
        st.error("⚠️ 'house_price_model.pkl' was not found! Please run your training script ('train_model.py') first in your command terminal.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            area = col1.number_input("Area (sqft):", int(df['Area_sqft'].min()), int(df['Area_sqft'].max()), 1500)
            beds = col2.number_input("Bedrooms Counts:", int(df['Bedrooms'].min()), int(df['Bedrooms'].max()), 3)
            baths = col3.number_input("Bathrooms Configuration Layouts:", int(df['Bathrooms'].min()), int(df['Bathrooms'].max()), 2)
            
            col4, col5, col6 = st.columns(3)
            age = col4.number_input("Age of Property (Years):", int(df['Age_Years'].min()), int(df['Age_Years'].max()), 5)
            dist = col5.number_input("Distance to City Center (km):", float(df['Distance_City_km'].min()), float(df['Distance_City_km'].max()), 5.0)
            schools = col6.number_input("Nearby Schools Counts:", int(df['Nearby_Schools'].min()), int(df['Nearby_Schools'].max()), 3)
            
            col7, col8 = st.columns(2)
            crime = col7.number_input("Crime Rate Scale:", float(df['Crime_Rate'].min()), float(df['Crime_Rate'].max()), 1.0)
            tax = col8.number_input("Annual Property Tax (PKR):", int(df['Property_Tax'].min()), int(df['Property_Tax'].max()), 20000)
            
            if st.form_submit_button("Predict Estimated Valuation"):
                payload = pd.DataFrame([[area, beds, baths, age, dist, schools, crime, tax]], 
                                       columns=["Area_sqft", "Bedrooms", "Bathrooms", "Age_Years", "Distance_City_km", "Nearby_Schools", "Crime_Rate", "Property_Tax"])
                prediction = model.predict(payload)[0]
                st.success(f"### Predicted Evaluation Price: PKR {prediction:,.2f}")

# --- PAGE 4: INVESTMENT ADVISOR ---
elif page == "Investment Advisor":
    st.title("📈 Capital Budget Optimizer")
    budget = st.number_input("Enter Maximum Purchasing Budget limit (PKR):", value=15000000, step=500000)
    matches = df[df["House_Price_PKR"] <= budget].sort_values(by="House_Price_PKR", ascending=False)
    st.subheader(f"Affordable Found Options Inventory ({len(matches)} matches)")
    st.dataframe(matches.head(25), use_container_width=True)

# --- PAGE 5: PROPERTY COMPARISON ---
elif page == "Property Comparison":
    st.title("⚖️ Side-by-Side Property Matrix Evaluator")
    p1 = st.selectbox("Choose Base Property Listing Index Reference A:", df.index.tolist(), index=0)
    p2 = st.selectbox("Choose Target Property Listing Index Reference B:", df.index.tolist(), index=min(1, len(df)-1))
    comparison_table = df.loc[[p1, p2]].T
    comparison_table.columns = [f"Property {p1}", f"Property {p2}"]
    st.table(comparison_table)

# --- PAGE 6: ANALYTICS ---
elif page == "Analytics":
    st.title("📊 Market Data Analytics Visualizations")
    fig1 = px.histogram(df, x="House_Price_PKR", title="Price Range Density Distributions", color_discrete_sequence=['#1E3A8A'])
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x="Area_sqft", y="House_Price_PKR", trendline="ols", title="Square Footage Structural Area vs Market Price", color_discrete_sequence=['#EF4444'])
    st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 7: DATASET EXPLORER ---
elif page == "Dataset Explorer":
    st.title("📂 Complete Multan Housing Datasets Hub")
    st.dataframe(df, use_container_width=True)
    st.subheader("General Mathematical Descriptive Summaries")
    st.dataframe(df.describe(), use_container_width=True)

# --- PAGE 8: ABOUT PROJECT ---
elif page == "About Project":
    st.title("ℹ️ Academic Framework Details")
    st.info("""
    **Project Title:** AI-Based Smart Real Estate Decision Support System  
    **Developer:** Khubaib Fakher  
    **Degree Program:** BS Artificial Intelligence (4th Semester)  
    **Core Stack Backend Engine:** Random Forest Regressor & Local FAISS Flat Indexing (TF-IDF Vector Space).
    """)

# -----------------------------------------------------------------------------
# 6. WHATSAPP SUPPORT FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📞 Direct Agent Consultations Channel")
st.markdown(
    """
    <a href="https://wa.me/923045285661" target="_blank">
        <button style="background-color:#25D366; color:white; border:none; padding:12px 28px; border-radius:6px; font-size:16px; font-weight:bold; cursor:pointer;">
            💬 Open Live Chat via WhatsApp Verification Office
        </button>
    </a>
    """,
    unsafe_allow_html=True
)