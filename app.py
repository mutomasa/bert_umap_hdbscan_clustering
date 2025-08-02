"""
BERT + UMAP + HDBSCAN Text Clustering Application
Streamlit-based web application for text clustering using BERT embeddings,
UMAP dimensionality reduction, and HDBSCAN clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from text_processor import TextProcessor
from bert_embedder import BERTEmbedder
from clustering_manager import ClusteringManager
from visualization_manager import VisualizationManager

# Page configuration
st.set_page_config(
    page_title="BERT + UMAP + HDBSCAN Text Clustering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)


class BERTClusteringApp:
    """BERT + UMAP + HDBSCAN Text Clustering Application"""
    
    def __init__(self):
        """Initialize the application"""
        self.text_processor = TextProcessor()
        self.bert_embedder = BERTEmbedder()
        self.clustering_manager = ClusteringManager()
        self.visualization_manager = VisualizationManager()
        
        # Initialize session state
        if 'embeddings' not in st.session_state:
            st.session_state.embeddings = None
        if 'clusters' not in st.session_state:
            st.session_state.clusters = None
        if 'umap_embeddings' not in st.session_state:
            st.session_state.umap_embeddings = None
        if 'texts' not in st.session_state:
            st.session_state.texts = None
    
    def run(self):
        """Run the application"""
        self._display_header()
        self._display_sidebar()
        self._display_main_content()
    
    def _display_header(self):
        """Display the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 BERT + UMAP + HDBSCAN Text Clustering</h1>
            <p>Advanced text clustering using BERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_sidebar(self):
        """Display the sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # BERT Model Selection
            st.subheader("🤖 BERT Model")
            bert_model = st.selectbox(
                "Select BERT Model",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "bert-base-uncased"
                ],
                index=0,
                help="Choose the BERT model for text embeddings"
            )
            
            # UMAP Parameters
            st.subheader("🗺️ UMAP Parameters")
            n_neighbors = st.slider(
                "Number of Neighbors",
                min_value=5,
                max_value=100,
                value=15,
                help="Number of neighbors for UMAP"
            )
            min_dist = st.slider(
                "Minimum Distance",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Minimum distance between points in UMAP"
            )
            n_components = st.slider(
                "Number of Components",
                min_value=2,
                max_value=10,
                value=2,
                help="Number of dimensions for UMAP reduction"
            )
            
            # HDBSCAN Parameters
            st.subheader("🔍 HDBSCAN Parameters")
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=2,
                max_value=50,
                value=5,
                help="Minimum number of samples in a cluster"
            )
            min_samples = st.slider(
                "Minimum Samples",
                min_value=1,
                max_value=20,
                value=3,
                help="Minimum number of samples in neighborhood"
            )
            
            # Save parameters to session state
            st.session_state.bert_model = bert_model
            st.session_state.n_neighbors = n_neighbors
            st.session_state.min_dist = min_dist
            st.session_state.n_components = n_components
            st.session_state.min_cluster_size = min_cluster_size
            st.session_state.min_samples = min_samples
    
    def _display_main_content(self):
        """Display the main content area"""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Text Input", 
            "🧠 Processing", 
            "📊 Visualization", 
            "📈 Analysis"
        ])
        
        with tab1:
            self._display_text_input_tab()
        
        with tab2:
            self._display_processing_tab()
        
        with tab3:
            self._display_visualization_tab()
        
        with tab4:
            self._display_analysis_tab()
    
    def _display_text_input_tab(self):
        """Display the text input tab"""
        st.header("📝 Text Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Sample Texts", "Upload CSV", "Manual Input", "URL Input"],
            horizontal=True
        )
        
        if input_method == "Sample Texts":
            self._load_sample_texts()
        elif input_method == "Upload CSV":
            self._load_csv_texts()
        elif input_method == "Manual Input":
            self._load_manual_texts()
        elif input_method == "URL Input":
            self._load_url_texts()
    
    def _load_sample_texts(self):
        """Load sample texts"""
        st.subheader("Sample Texts")
        
        sample_categories = {
            "Technology": [
                "Artificial intelligence is transforming industries worldwide.",
                "Machine learning algorithms are becoming more sophisticated.",
                "Deep learning has revolutionized computer vision.",
                "Natural language processing enables better communication.",
                "Data science is essential for modern business decisions.",
                "Cloud computing provides scalable infrastructure.",
                "Cybersecurity protects against digital threats.",
                "Blockchain technology ensures secure transactions.",
                "Internet of Things connects devices globally.",
                "Virtual reality creates immersive experiences."
            ],
            "Science": [
                "Quantum physics challenges our understanding of reality.",
                "Climate change affects global ecosystems.",
                "Genetic engineering advances medical treatments.",
                "Astronomy reveals the universe's mysteries.",
                "Chemistry explains molecular interactions.",
                "Biology studies living organisms.",
                "Physics describes fundamental forces.",
                "Mathematics provides logical frameworks.",
                "Geology examines Earth's structure.",
                "Psychology explores human behavior."
            ],
            "Business": [
                "Marketing strategies drive customer engagement.",
                "Financial planning ensures long-term success.",
                "Human resources manage employee relations.",
                "Operations management optimizes processes.",
                "Strategic planning guides organizational growth.",
                "Customer service builds brand loyalty.",
                "Product development creates market value.",
                "Sales techniques increase revenue.",
                "Supply chain management reduces costs.",
                "Leadership skills inspire team performance."
            ]
        }
        
        selected_category = st.selectbox(
            "Select sample category:",
            list(sample_categories.keys())
        )
        
        if selected_category:
            texts = sample_categories[selected_category]
            st.session_state.texts = texts
            
            # Display texts
            st.write(f"**Loaded {len(texts)} texts from {selected_category} category:**")
            for i, text in enumerate(texts, 1):
                st.write(f"{i}. {text}")
            
            st.success(f"✅ Loaded {len(texts)} sample texts successfully!")
    
    def _load_csv_texts(self):
        """Load texts from CSV file"""
        st.subheader("Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'text' column containing your documents"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                    st.session_state.texts = texts
                    st.success(f"✅ Loaded {len(texts)} texts from CSV successfully!")
                    
                    # Display sample
                    st.write("**Sample texts:**")
                    for i, text in enumerate(texts[:5], 1):
                        st.write(f"{i}. {text}")
                    
                    if len(texts) > 5:
                        st.write(f"... and {len(texts) - 5} more texts")
                else:
                    st.error("❌ CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"❌ Error loading CSV file: {str(e)}")
    
    def _load_manual_texts(self):
        """Load manually entered texts"""
        st.subheader("Manual Text Input")
        
        text_input = st.text_area(
            "Enter your texts (one per line):",
            height=300,
            placeholder="Enter your texts here, one per line...\n\nExample:\nThis is the first text.\nThis is the second text.\nThis is the third text."
        )
        
        if text_input:
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            st.session_state.texts = texts
            st.success(f"✅ Loaded {len(texts)} texts successfully!")
    
    def _load_url_texts(self):
        """Load texts from URL"""
        st.subheader("URL Text Input")
        
        url = st.text_input(
            "Enter URL to fetch texts:",
            placeholder="https://example.com/texts.txt"
        )
        
        if url:
            try:
                import requests
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                texts = [line.strip() for line in response.text.split('\n') if line.strip()]
                st.session_state.texts = texts
                st.success(f"✅ Loaded {len(texts)} texts from URL successfully!")
            except Exception as e:
                st.error(f"❌ Error loading texts from URL: {str(e)}")
    
    def _display_processing_tab(self):
        """Display the processing tab"""
        st.header("🧠 Processing")
        
        if st.session_state.texts is None:
            st.info("👆 Please load texts in the 'Text Input' tab first.")
            return
        
        # Display processing options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Processing", type="primary"):
                self._process_texts()
        
        with col2:
            if st.button("🔄 Reset Processing"):
                st.session_state.embeddings = None
                st.session_state.clusters = None
                st.session_state.umap_embeddings = None
                st.success("✅ Processing reset successfully!")
        
        # Display processing status
        if st.session_state.embeddings is not None:
            st.success("✅ BERT embeddings generated!")
            
        if st.session_state.umap_embeddings is not None:
            st.success("✅ UMAP dimensionality reduction completed!")
            
        if st.session_state.clusters is not None:
            st.success("✅ HDBSCAN clustering completed!")
            
            # Display clustering statistics
            self._display_clustering_stats()
    
    def _process_texts(self):
        """Process the texts through the pipeline"""
        with st.spinner("🔄 Processing texts..."):
            try:
                # Step 1: Generate BERT embeddings
                st.write("**Step 1: Generating BERT embeddings...**")
                embeddings = self.bert_embedder.generate_embeddings(
                    st.session_state.texts,
                    model_name=st.session_state.bert_model
                )
                st.session_state.embeddings = embeddings
                
                # Step 2: UMAP dimensionality reduction
                st.write("**Step 2: Performing UMAP dimensionality reduction...**")
                umap_embeddings = self.clustering_manager.apply_umap(
                    embeddings,
                    n_neighbors=st.session_state.n_neighbors,
                    min_dist=st.session_state.min_dist,
                    n_components=st.session_state.n_components
                )
                st.session_state.umap_embeddings = umap_embeddings
                
                # Step 3: HDBSCAN clustering
                st.write("**Step 3: Performing HDBSCAN clustering...**")
                clusters = self.clustering_manager.apply_hdbscan(
                    umap_embeddings,
                    min_cluster_size=st.session_state.min_cluster_size,
                    min_samples=st.session_state.min_samples
                )
                st.session_state.clusters = clusters
                
                st.success("🎉 Processing completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
    
    def _display_clustering_stats(self):
        """Display clustering statistics"""
        if st.session_state.clusters is None:
            return
        
        st.subheader("📊 Clustering Statistics")
        
        # Calculate statistics
        n_clusters = len(set(st.session_state.clusters)) - (1 if -1 in st.session_state.clusters else 0)
        n_noise = list(st.session_state.clusters).count(-1)
        total_points = len(st.session_state.clusters)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Texts", total_points)
        
        with col2:
            st.metric("Number of Clusters", n_clusters)
        
        with col3:
            st.metric("Noise Points", n_noise)
        
        with col4:
            if total_points > 0:
                cluster_ratio = (total_points - n_noise) / total_points * 100
                st.metric("Clustered Ratio", f"{cluster_ratio:.1f}%")
    
    def _display_visualization_tab(self):
        """Display the visualization tab"""
        st.header("📊 Visualization")
        
        if st.session_state.umap_embeddings is None or st.session_state.clusters is None:
            st.info("👆 Please complete processing in the 'Processing' tab first.")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Select visualization type:",
            ["UMAP 2D Scatter Plot", "UMAP 3D Scatter Plot", "Cluster Distribution", "Word Clouds"]
        )
        
        if viz_type == "UMAP 2D Scatter Plot":
            self._display_umap_2d_plot()
        elif viz_type == "UMAP 3D Scatter Plot":
            self._display_umap_3d_plot()
        elif viz_type == "Cluster Distribution":
            self._display_cluster_distribution()
        elif viz_type == "Word Clouds":
            self._display_word_clouds()
    
    def _display_umap_2d_plot(self):
        """Display UMAP 2D scatter plot"""
        st.subheader("UMAP 2D Scatter Plot")
        
        fig = self.visualization_manager.create_umap_2d_plot(
            st.session_state.umap_embeddings,
            st.session_state.clusters,
            st.session_state.texts
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_umap_3d_plot(self):
        """Display UMAP 3D scatter plot"""
        st.subheader("UMAP 3D Scatter Plot")
        
        if st.session_state.umap_embeddings.shape[1] < 3:
            st.warning("⚠️ 3D plot requires at least 3 UMAP components. Please increase 'Number of Components' in the sidebar.")
            return
        
        fig = self.visualization_manager.create_umap_3d_plot(
            st.session_state.umap_embeddings,
            st.session_state.clusters,
            st.session_state.texts
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_cluster_distribution(self):
        """Display cluster distribution"""
        st.subheader("Cluster Distribution")
        
        fig = self.visualization_manager.create_cluster_distribution_plot(
            st.session_state.clusters
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_word_clouds(self):
        """Display word clouds for each cluster"""
        st.subheader("Word Clouds by Cluster")
        
        if st.session_state.texts is None or st.session_state.clusters is None:
            return
        
        # Get unique clusters (excluding noise)
        unique_clusters = sorted(set(st.session_state.clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        # Create word clouds for each cluster
        for cluster_id in unique_clusters:
            cluster_texts = [
                text for text, cluster in zip(st.session_state.texts, st.session_state.clusters)
                if cluster == cluster_id
            ]
            
            if cluster_texts:
                st.write(f"**Cluster {cluster_id}** ({len(cluster_texts)} texts)")
                
                # Create word cloud
                wordcloud = self.visualization_manager.create_wordcloud(cluster_texts)
                
                if wordcloud is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Show sample texts
                with st.expander(f"Show sample texts from Cluster {cluster_id}"):
                    for i, text in enumerate(cluster_texts[:5], 1):
                        st.write(f"{i}. {text}")
                    if len(cluster_texts) > 5:
                        st.write(f"... and {len(cluster_texts) - 5} more texts")
    
    def _display_analysis_tab(self):
        """Display the analysis tab"""
        st.header("📈 Analysis")
        
        if st.session_state.clusters is None:
            st.info("👆 Please complete processing in the 'Processing' tab first.")
            return
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Cluster Analysis", "Text Similarity", "Topic Modeling", "Export Results"]
        )
        
        if analysis_type == "Cluster Analysis":
            self._display_cluster_analysis()
        elif analysis_type == "Text Similarity":
            self._display_text_similarity()
        elif analysis_type == "Topic Modeling":
            self._display_topic_modeling()
        elif analysis_type == "Export Results":
            self._display_export_results()
    
    def _display_cluster_analysis(self):
        """Display detailed cluster analysis"""
        st.subheader("Cluster Analysis")
        
        if st.session_state.texts is None or st.session_state.clusters is None:
            return
        
        # Create analysis dataframe
        df = pd.DataFrame({
            'text': st.session_state.texts,
            'cluster': st.session_state.clusters
        })
        
        # Display cluster details
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            if cluster_id == -1:
                st.write(f"**Noise Cluster** ({len(cluster_data)} texts)")
            else:
                st.write(f"**Cluster {cluster_id}** ({len(cluster_data)} texts)")
            
            # Show cluster statistics
            avg_length = cluster_data['text'].str.len().mean()
            st.write(f"Average text length: {avg_length:.1f} characters")
            
            # Show sample texts
            with st.expander(f"Show texts from {'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'}"):
                for i, text in enumerate(cluster_data['text'].head(10), 1):
                    st.write(f"{i}. {text}")
    
    def _display_text_similarity(self):
        """Display text similarity analysis"""
        st.subheader("Text Similarity Analysis")
        
        if st.session_state.embeddings is None:
            st.info("Please generate embeddings first.")
            return
        
        # Select texts to compare
        text1_idx = st.selectbox(
            "Select first text:",
            range(len(st.session_state.texts)),
            format_func=lambda x: st.session_state.texts[x][:50] + "..." if len(st.session_state.texts[x]) > 50 else st.session_state.texts[x]
        )
        
        text2_idx = st.selectbox(
            "Select second text:",
            range(len(st.session_state.texts)),
            format_func=lambda x: st.session_state.texts[x][:50] + "..." if len(st.session_state.texts[x]) > 50 else st.session_state.texts[x]
        )
        
        if text1_idx != text2_idx:
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity = cosine_similarity(
                [st.session_state.embeddings[text1_idx]],
                [st.session_state.embeddings[text2_idx]]
            )[0][0]
            
            st.write(f"**Cosine Similarity:** {similarity:.4f}")
            
            # Display texts
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Text 1:**")
                st.write(st.session_state.texts[text1_idx])
            with col2:
                st.write("**Text 2:**")
                st.write(st.session_state.texts[text2_idx])
    
    def _display_topic_modeling(self):
        """Display topic modeling analysis"""
        st.subheader("Topic Modeling Analysis")
        
        if st.session_state.texts is None or st.session_state.clusters is None:
            return
        
        # Simple topic modeling using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Prepare texts for topic modeling
        texts = st.session_state.texts
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply LDA
        n_topics = st.slider("Number of topics:", 2, 10, 5)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
        
        lda.fit(tfidf_matrix)
        
        # Display topics
        feature_names = vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
    
    def _display_export_results(self):
        """Display export results functionality"""
        st.subheader("Export Results")
        
        if st.session_state.texts is None or st.session_state.clusters is None:
            return
        
        # Create results dataframe
        df = pd.DataFrame({
            'text': st.session_state.texts,
            'cluster': st.session_state.clusters
        })
        
        if st.session_state.umap_embeddings is not None:
            for i in range(st.session_state.umap_embeddings.shape[1]):
                df[f'umap_component_{i+1}'] = st.session_state.umap_embeddings[:, i]
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv"
        )
        
        # Display results table
        st.write("**Results Preview:**")
        st.dataframe(df.head(10))


def main():
    """Main function"""
    try:
        app = BERTClusteringApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main() 