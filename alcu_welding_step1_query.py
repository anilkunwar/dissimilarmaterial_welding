import arxiv
import pandas as pd
import streamlit as st
import urllib.request
import os
from datetime import datetime
import time

# Initialize Streamlit app
st.set_page_config(page_title="Al-Cu Welding arXiv Explorer", layout="wide")
st.title("Al-Cu Dissimilar Welding Parameters Explorer (arXiv)")
st.markdown("""
Welcome to the Al-Cu Dissimilar Welding Parameters Explorer! This tool queries arXiv to find papers related to aluminum-copper (Al-Cu) dissimilar laser welding. Use the options below to customize your search, download PDFs, and view results. If no papers are found, try broadening the query or adjusting categories.
""")

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Query arXiv function
def query_arxiv(query, categories, max_results, start_year, end_year):
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded"
                })
            if len(papers) >= max_results:
                break
        return papers
    except Exception as e:
        st.error(f"Error querying arXiv: {e}")
        return []

# Download PDF function
def download_pdf(pdf_url, paper_id):
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
        return f"Downloaded ({file_size:.1f} KB)"
    except Exception as e:
        return f"Failed: {str(e)}"

# Sidebar for search inputs
with st.sidebar:
    st.header("Search Options")
    
    # Query input
    query_option = st.radio(
        "Select Query Type",
        ["Default Query", "Custom Query", "Suggested Queries"],
        help="Choose how to specify the search query."
    )
    if query_option == "Default Query":
        query = "aluminum copper dissimilar welding process parameters"
        st.write("Using default query: **" + query + "**")
    elif query_option == "Custom Query":
        query = st.text_input("Enter Custom Query", value="aluminum copper dissimilar welding")
        st.write("Custom query: **" + query + "**")
    else:
        suggested_queries = [
            "laser welding aluminum copper",
            "Al-Cu dissimilar welding",
            "multimaterial welding Al-Cu",
            "laser welding process parameters"
        ]
        query = st.selectbox("Choose Suggested Query", suggested_queries)
        st.write("Selected query: **" + query + "**")
    
    # Categories input
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph"]
    extra_categories = ["physics.optics", "cond-mat.other"]
    categories = st.multiselect(
        "Select arXiv Categories",
        default_categories + extra_categories,
        default=default_categories,
        help="Choose categories to filter papers (e.g., materials science, applied physics)."
    )
    
    # Max results
    max_results = st.slider(
        "Maximum Number of Papers",
        min_value=1,
        max_value=50,
        value=10,
        help="Set the maximum number of papers to retrieve."
    )
    
    # Year range
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=1900,
            max_value=current_year,
            value=2015,
            help="Earliest publication year."
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=start_year,
            max_value=current_year,
            value=current_year,
            help="Latest publication year."
        )
    
    # Search button
    search_button = st.button("Search arXiv")

# Main content
st.header("Search Results")

if search_button:
    if not query.strip():
        st.error("Please enter a valid query.")
    elif not categories:
        st.error("Please select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be less than or equal to end year.")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found matching your criteria.")
            st.markdown("""
            **Suggestions to find more papers:**
            - Broaden the query (e.g., try 'laser welding aluminum copper' or 'Al-Cu welding').
            - Add more categories (e.g., 'physics.optics' for laser-related papers).
            - Expand the year range (e.g., 2010–2025).
            - Increase the maximum number of papers.
            """)
        else:
            st.success(f"Found **{len(papers)}** papers matching your query!")
            st.write(f"Query: **{query}** | Categories: **{', '.join(categories)}** | Years: **{start_year}–{end_year}**")
            
            # Download PDFs
            st.subheader("Downloading PDFs")
            progress_bar = st.progress(0)
            for i, paper in enumerate(papers):
                if paper["pdf_url"]:
                    status = download_pdf(paper["pdf_url"], paper["id"])
                    paper["download_status"] = status
                else:
                    paper["download_status"] = "No PDF URL"
                progress_bar.progress((i + 1) / len(papers))
                time.sleep(0.1)  # Avoid overwhelming arXiv servers
            
            # Display results
            df = pd.DataFrame(papers)
            st.subheader("Paper Details")
            st.dataframe(
                df[["id", "title", "year", "categories", "abstract", "download_status"]],
                use_container_width=True
            )
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Paper Metadata CSV",
                csv,
                "alcu_papers_metadata.csv",
                "text/csv"
            )
            
            # Summary
            downloaded = sum(1 for p in papers if "Downloaded" in p["download_status"])
            st.write(f"**Summary**: {len(papers)} papers found, {downloaded} PDFs downloaded successfully.")
            if downloaded < len(papers):
                st.warning("Some PDFs failed to download. Check 'download_status' for details.")

else:
    st.info("Use the sidebar to configure your search and click 'Search arXiv' to begin.")

# Footer
st.markdown("---")
st.write("Developed for Al-Cu dissimilar welding research. Step 1: arXiv query and PDF download.")