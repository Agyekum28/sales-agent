# Sales Assistant Agent using Tavily Search and Groq LLM
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
from langchain_core.caches import InMemoryCache
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
from dotenv import load_dotenv
# Set up caching for LLM responses
set_llm_cache(InMemoryCache())
# Load environment variables from .env file
load_dotenv()
# LLM and Search Tool Initialization
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # ✅ valid model name
    temperature=0.01,                  # ✅ set temp as a separate argument
    max_retries=3,
    timeout=120
)


search_tool = TavilySearch(topic="general", max_results=2)

results = search_tool.invoke({"query": "example query"})
print("Result: ", results)
def generate_insights(company_url, product_name, competitors, pdf_text):

    # Perform web search to gather information
    search_query = f"Company: {company_url}, Product: {product_name}, Competitors: {competitors}"
    search_results = search_tool.invoke({"query": search_query})
    messages = [
        SystemMessage(f"You're a helpful sales assistant with over 20 years of calculated consumer sales experiance. answer the user question using the provided content: {results}"),
        HumanMessage(content=f"""
         PDF Content: {pdf_text}    

        Company Info from Tavily: {search_results}
        
        Product: {product_name}
        Competitors: {competitors}
        
        Generate a one-page report including the following sections, in addition to any other relevant insights like recent news, financial data, or market trends, and also relaying a prefered pathway to entry for a new product/service in the market that meets reasonable demand:
        1. Company strategy related to {product_name}
        2. Possible competitors or partnerships in this area
        3. Leadership and decision-makers relevant to this area
        Format output in clear sections with bullet points.
        """)
    ]

    model_response = llm.invoke(messages)
    print("\n Model Response: ", model_response.content)
    return model_response.content

# UI with Streamlit

st.title("Sales Assistant Agent")
st.subheader("Intelligent Assistance for Modern Sales Teams")
st.divider()
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Combine PDF text into a single string
    pdf_text = "\n".join([doc.page_content for doc in documents])

    st.session_state["pdf_text"] = pdf_text   # store it for later
    st.success("PDF uploaded successfully!")

st.divider()
company_name = st.text_input('Company Name')
company_url = st.text_input("Company Website URL:")
product_name = st.text_input("Product/Service Name:")
competitors = st.text_input("Known Competitors")

if st.button("Generate Insights"):
    if uploaded_file or (company_url and company_name):
        with st.spinner("Generating report..."):
            pdf_text = st.session_state.get("pdf_text", "")
            insights = generate_insights(company_url, product_name, competitors, pdf_text)
            st.subheader("Account Insights")
            st.write(insights)
    else:
        st.warning("Please provide at least a Company URL and Company Name.")





