import streamlit as st
import pandas as pd
from vector_store import RealEstateVectorStore

def render_training_hub(vector_store: RealEstateVectorStore):
    st.title("🤖 AI & LLM Training Dataset Explorer")
    st.markdown("### Client & Property Dealer Conversations (Roman English)")
    st.write("This module manages the **Expert Q&A Pairs** compiled for fine-tuning Large Language Models or intent classification frameworks.")
    
    # Synchronization tracking metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Q&A Conversations", len(vector_store.df))
    col2.metric("Unique Domain Categories", len(vector_store.df["Category"].unique()) if not vector_store.df.empty else 0)
    col3.info("System Structure format: Prompt (Question) ➔ Completion (Answer)")

    # Administrator Interface panel update section
    with st.expander("➕ Add New Expert Knowledge Base Pair (Teacher / Admin Panel)"):
        with st.form("new_qa_form", clear_on_submit=True):
            cat_options = ["General Inquiry", "Location & Amenities", "Pricing & Budgeting", "Investment & ROI", "Legal & Documentation", "Property Features", "Dealer Commission & Process", "Negotiation & Token Money"]
            new_cat = st.selectbox("Select Domain Classification category:", cat_options)
            new_q = st.text_input("Enter Client Question Prompt (Roman English):")
            new_a = st.text_area("Enter Expert Dealer Answer Completion:")
            
            submit_btn = st.form_submit_button("Inject to Vector Database")
            if submit_btn:
                if new_q.strip() and new_a.strip():
                    vector_store.append_data(new_cat, new_q.strip(), new_a.strip())
                    st.success("🎉 Record embedded and FAISS indexing synchronized!")
                    st.rerun()
                else:
                    st.error("Both fields are strictly required to compute vectors.")

    # Main Data Explorer Layout view 
    st.subheader("🔍 Vector Matrix Data Collection Browser")
    if not vector_store.df.empty:
        selected_cat = st.selectbox("Filter Category Matrix:", ["All Topics"] + list(vector_store.df["Category"].unique()))
        search_kw = st.text_input("Search content via exact keyword sequence strings:")

        filtered_df = vector_store.df.copy()
        if selected_cat != "All Topics":
            filtered_df = filtered_df[filtered_df["Category"] == selected_cat]
        if search_kw:
            filtered_df = filtered_df[
                filtered_df["Question"].str.contains(search_kw, case=False, na=False) |
                filtered_df["Answer"].str.contains(search_kw, case=False, na=False)
            ]

        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("Vector database table pipeline is completely clean and empty.")