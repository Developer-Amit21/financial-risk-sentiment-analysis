
import streamlit as st
import pandas as pd
import sqlite3
import json # Import json to parse the JSON strings from the database
import altair as alt # Import altair for potential visualizations
from pathlib import Path

# --- Database Functions ---
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "news.db"

def get_all_news_df():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Select all columns, including the new ones
        df = pd.read_sql_query("SELECT * FROM news", conn)
        # Ensure the DataFrame has the expected columns, including the new ones
        # This helps if the database table was created with an older schema
        expected_cols = [
            'id', 'title', 'description', 'url', 'published_at', 'source',
            'sentiment', 'risk', 'title_lang', 'title_en', 'description_lang',
            'description_en', 'credibility_score', 'entities', 'relations',
            'events', 'topic', 'topic_probability', 'sentiment_score'
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None # Add missing column with None values

        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- Filtering Function ---
def filter_news(df, search_query, min_credibility):
    """
    Filters news articles based on search query and minimum credibility score.
    """
    filtered_df = df.copy()

    # Apply search query filter
    if search_query:
        query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['title'].str.lower().str.contains(query, na=False) |
            filtered_df['description'].str.lower().str.contains(query, na=False) |
            filtered_df['title_en'].str.lower().str.contains(query, na=False) | # Search translated titles
            filtered_df['description_en'].str.lower().str.contains(query, na=False) # Search translated descriptions
        ]

    # Apply credibility filter
    # Handle potential None or non-numeric values in 'credibility_score'
    filtered_df = filtered_df[pd.to_numeric(filtered_df['credibility_score'], errors='coerce').fillna(0) >= min_credibility]


    return filtered_df

# --- UI ---
st.set_page_config(page_title="Risk-Aware News", layout="wide")
st.title("📈 Risk-Aware Financial News Dashboard")

# --- Load Data ---
news_df = get_all_news_df()

if news_df.empty:
    st.warning("No news found in the database. Please run the collection pipeline.")
else:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Credibility Score Slider
    min_credibility = st.sidebar.slider(
        "Minimum Credibility Score",
        min_value=0.0,
        max_value=1.0,
        value=0.4, # Default value set in the original credibility_score function
        step=0.05,
        help="Filter news articles by a minimum credibility score (0.0 to 1.0)."
    )

    # Add other potential filters here later (e.g., Risk Level, Sentiment, Topic)
    # risk_filter = st.sidebar.multiselect("Filter by Risk Level", news_df['risk'].unique())
    # sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", news_df['sentiment'].unique())
    # topic_filter = st.sidebar.multiselect("Filter by Topic", news_df['topic'].unique())


    # --- Main Content ---
    # Search Bar
    search_query = st.text_input("Search news articles:", "")


    # --- Filtering ---
    filtered_df = filter_news(news_df, search_query, min_credibility)

    st.write(f"Displaying {len(filtered_df)} of {len(news_df)} articles based on filters.")

    # --- Optional: Visualizations ---
    # Example: Distribution of Sentiment for filtered news
    if not filtered_df.empty:
        st.subheader("Filtered News Analysis")

        sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        chart = alt.Chart(sentiment_counts).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment',
            tooltip=['Sentiment', 'Count']
        ).properties(
            title='Sentiment Distribution of Filtered News'
        )
        st.altair_chart(chart, use_container_width=True)

        # Example: Distribution of Risk Levels for filtered news
        if 'risk' in filtered_df.columns and not filtered_df['risk'].isnull().all():
             risk_counts = filtered_df['risk'].value_counts().reset_index()
             risk_counts.columns = ['Risk Level', 'Count']
             risk_chart = alt.Chart(risk_counts).mark_bar().encode(
                 x='Risk Level',
                 y='Count',
                 color='Risk Level',
                 tooltip=['Risk Level', 'Count']
             ).properties(
                 title='Risk Level Distribution of Filtered News'
             )
             st.altair_chart(risk_chart, use_container_width=True)

        # Example: Distribution of Topics (only if topic column exists and has data)
        if 'topic' in filtered_df.columns and not filtered_df['topic'].isnull().all():
             # You might want to map topic IDs to representative words for better display
             # This requires access to the topic_info generated by BERTopic, which is not saved in the DB.
             # For simplicity, we'll show topic IDs.
             topic_counts = filtered_df['topic'].value_counts().reset_index()
             topic_counts.columns = ['Topic ID', 'Count']
             # Convert topic ID to string for categorical plotting
             topic_counts['Topic ID'] = topic_counts['Topic ID'].astype(str)
             topic_chart = alt.Chart(topic_counts).mark_bar().encode(
                 x=alt.X('Topic ID', sort='-y'), # Sort by count
                 y='Count',
                 color='Topic ID',
                 tooltip=['Topic ID', 'Count']
             ).properties(
                 title='Topic Distribution of Filtered News (by ID)'
             ).interactive() # Make it zoomable/pannable if many topics
             st.altair_chart(topic_chart, use_container_width=True)



    st.subheader("News Articles")
    # --- Display News ---
    # Limit to top 50 for display to keep the page manageable
    for index, row in filtered_df.head(50).iterrows():
        st.subheader(row['title'])
        st.caption(f"Source: {row['source']} | Published: {row['published_at']}")

        # Display sentiment, risk, and credibility
        st.write(f"**Sentiment:** {row['sentiment']} (Score: {row['sentiment_score']:.2f}) | **Risk:** {row['risk']} | **Credibility Score:** {row['credibility_score']:.2f}")

        # Display Topic
        if 'topic' in row and row['topic'] is not None and row['topic'] != -1:
             st.write(f"**Topic:** ID {row['topic']} (Probability: {row['topic_probability']:.2f})")
             # In a real app, you might display representative words for the topic ID

        # Display original and translated description
        st.write(f"**Original Language:** {row['description_lang']}")
        st.write(f"**Description:** {row['description']}")
        if row['description_lang'] != 'en':
             st.write(f"**Translated Description (en):** {row['description_en']}")


        # Display extracted NLP features (Entities, Relations, Events)
        st.write("**Extracted Information:**")
        try:
            entities = json.loads(row['entities']) if row['entities'] else []
            relations = json.loads(row['relations']) if row['relations'] else []
            events = json.loads(row['events']) if row['events'] else []

            if entities:
                 st.write("- **Entities:** " + ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities]))
            if relations:
                 # Display relations in a more readable format
                 relation_strs = [f"{rel['head']} --[{rel['relation']}]--> {rel['tail']}" for rel in relations if 'head' in rel and 'relation' in rel]
                 if relation_strs:
                    st.write("- **Relations:** " + ", ".join(relation_strs))
                 else:
                    st.write("- **Relations:** None identified")
            if events:
                 st.write("- **Events:** " + ", ".join(events))

            if not entities and not relations and not events:
                 st.write("  None")

        except json.JSONDecodeError:
            st.write("  Error parsing extracted information.")
            st.write(f"  Entities Raw: {row['entities']}")
            st.write(f"  Relations Raw: {row['relations']}")
            st.write(f"  Events Raw: {row['events']}")


        st.markdown(f"[Read more]({row['url']})", unsafe_allow_html=True)
        st.markdown("---")

