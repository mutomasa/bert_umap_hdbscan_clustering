"""
Visualization Manager Module
Handles visualization of clustering results using matplotlib and plotly.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import List, Optional, Dict, Any
import pandas as pd
from collections import Counter
import re


class VisualizationManager:
    """Visualization manager for clustering results"""
    
    def __init__(self):
        """Initialize the visualization manager"""
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color palette for clusters
        self.color_palette = px.colors.qualitative.Set3
    
    def create_umap_2d_plot(
        self, 
        umap_embeddings: np.ndarray, 
        cluster_labels: np.ndarray, 
        texts: List[str] = None
    ) -> go.Figure:
        """
        Create 2D UMAP scatter plot
        
        Args:
            umap_embeddings: UMAP reduced embeddings
            cluster_labels: Cluster labels
            texts: Original texts for hover information
            
        Returns:
            Plotly figure
        """
        # Create dataframe for plotting
        df = pd.DataFrame({
            'UMAP1': umap_embeddings[:, 0],
            'UMAP2': umap_embeddings[:, 1],
            'Cluster': cluster_labels
        })
        
        if texts:
            df['Text'] = [text[:100] + '...' if len(text) > 100 else text for text in texts]
        
        # Create figure
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='Cluster',
            hover_data=['Text'] if texts else None,
            title='UMAP 2D Visualization of Text Clusters',
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='UMAP Component 1',
            yaxis_title='UMAP Component 2',
            legend_title='Cluster',
            width=800,
            height=600
        )
        
        # Update traces
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            selector=dict(mode='markers')
        )
        
        return fig
    
    def create_umap_3d_plot(
        self, 
        umap_embeddings: np.ndarray, 
        cluster_labels: np.ndarray, 
        texts: List[str] = None
    ) -> go.Figure:
        """
        Create 3D UMAP scatter plot
        
        Args:
            umap_embeddings: UMAP reduced embeddings (3D)
            cluster_labels: Cluster labels
            texts: Original texts for hover information
            
        Returns:
            Plotly figure
        """
        # Create dataframe for plotting
        df = pd.DataFrame({
            'UMAP1': umap_embeddings[:, 0],
            'UMAP2': umap_embeddings[:, 1],
            'UMAP3': umap_embeddings[:, 2],
            'Cluster': cluster_labels
        })
        
        if texts:
            df['Text'] = [text[:100] + '...' if len(text) > 100 else text for text in texts]
        
        # Create figure
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='Cluster',
            hover_data=['Text'] if texts else None,
            title='UMAP 3D Visualization of Text Clusters',
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            scene=dict(
                xaxis_title='UMAP Component 1',
                yaxis_title='UMAP Component 2',
                zaxis_title='UMAP Component 3'
            ),
            legend_title='Cluster',
            width=800,
            height=600
        )
        
        # Update traces
        fig.update_traces(
            marker=dict(size=4, opacity=0.7),
            selector=dict(mode='markers')
        )
        
        return fig
    
    def create_cluster_distribution_plot(self, cluster_labels: np.ndarray) -> go.Figure:
        """
        Create cluster distribution plot
        
        Args:
            cluster_labels: Cluster labels
            
        Returns:
            Plotly figure
        """
        # Count cluster sizes
        cluster_counts = Counter(cluster_labels)
        
        # Create dataframe
        df = pd.DataFrame({
            'Cluster': list(cluster_counts.keys()),
            'Count': list(cluster_counts.values())
        })
        
        # Sort by cluster label
        df = df.sort_values('Cluster')
        
        # Create figure
        fig = px.bar(
            df,
            x='Cluster',
            y='Count',
            title='Cluster Size Distribution',
            color='Cluster',
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Cluster ID',
            yaxis_title='Number of Texts',
            showlegend=False,
            width=800,
            height=500
        )
        
        # Add text labels on bars
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )
        
        return fig
    
    def create_wordcloud(self, texts: List[str], max_words: int = 100) -> Optional[WordCloud]:
        """
        Create word cloud from texts
        
        Args:
            texts: List of texts
            max_words: Maximum number of words to include
            
        Returns:
            WordCloud object or None
        """
        if not texts:
            return None
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Clean text for word cloud
        cleaned_text = re.sub(r'[^\w\s]', '', combined_text.lower())
        
        if not cleaned_text.strip():
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(cleaned_text)
        
        return wordcloud
    
    def create_similarity_heatmap(
        self, 
        similarity_matrix: np.ndarray, 
        texts: List[str] = None
    ) -> go.Figure:
        """
        Create similarity heatmap
        
        Args:
            similarity_matrix: Similarity matrix
            texts: Text labels for axes
            
        Returns:
            Plotly figure
        """
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='Viridis',
            showscale=True,
            text=similarity_matrix.round(3),
            texttemplate="%{text}",
            textfont={"size": 8}
        ))
        
        # Update layout
        fig.update_layout(
            title='Text Similarity Matrix',
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Text Index',
            yaxis_title='Text Index',
            width=700,
            height=600
        )
        
        return fig
    
    def create_cluster_metrics_plot(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create cluster metrics plot
        
        Args:
            metrics: Dictionary of clustering metrics
            
        Returns:
            Plotly figure
        """
        if not metrics:
            return go.Figure()
        
        # Create dataframe
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Score': list(metrics.values())
        })
        
        # Create figure
        fig = px.bar(
            df,
            x='Metric',
            y='Score',
            title='Clustering Quality Metrics',
            color='Score',
            color_continuous_scale='viridis',
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Metric',
            yaxis_title='Score',
            width=600,
            height=400
        )
        
        return fig
    
    def create_text_length_distribution(self, texts: List[str]) -> go.Figure:
        """
        Create text length distribution plot
        
        Args:
            texts: List of texts
            
        Returns:
            Plotly figure
        """
        # Calculate text lengths
        text_lengths = [len(text.split()) for text in texts]
        
        # Create figure
        fig = px.histogram(
            x=text_lengths,
            title='Text Length Distribution',
            nbins=20,
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Number of Words',
            yaxis_title='Frequency',
            width=600,
            height=400
        )
        
        return fig
    
    def create_cluster_comparison_plot(
        self, 
        cluster_labels: np.ndarray, 
        texts: List[str]
    ) -> go.Figure:
        """
        Create cluster comparison plot showing text characteristics
        
        Args:
            cluster_labels: Cluster labels
            texts: Original texts
            
        Returns:
            Plotly figure
        """
        # Calculate text characteristics
        text_lengths = [len(text.split()) for text in texts]
        
        # Create dataframe
        df = pd.DataFrame({
            'Cluster': cluster_labels,
            'Text_Length': text_lengths,
            'Text': texts
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Text Length by Cluster', 'Cluster Size Distribution', 
                          'Text Length Distribution', 'Cluster Statistics'),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Box plot of text lengths by cluster
        for cluster in sorted(set(cluster_labels)):
            cluster_data = df[df['Cluster'] == cluster]['Text_Length']
            fig.add_trace(
                go.Box(y=cluster_data, name=f'Cluster {cluster}', showlegend=False),
                row=1, col=1
            )
        
        # Bar plot of cluster sizes
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=cluster_counts.index, y=cluster_counts.values, name='Cluster Size'),
            row=1, col=2
        )
        
        # Histogram of text lengths
        fig.add_trace(
            go.Histogram(x=text_lengths, name='Text Length'),
            row=2, col=1
        )
        
        # Scatter plot of cluster vs text length
        fig.add_trace(
            go.Scatter(x=cluster_labels, y=text_lengths, mode='markers', name='Text Length'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Cluster Analysis Dashboard',
            title_x=0.5,
            title_font_size=16,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_cluster_plot(
        self, 
        umap_embeddings: np.ndarray, 
        cluster_labels: np.ndarray, 
        texts: List[str]
    ) -> go.Figure:
        """
        Create interactive cluster plot with detailed hover information
        
        Args:
            umap_embeddings: UMAP reduced embeddings
            cluster_labels: Cluster labels
            texts: Original texts
            
        Returns:
            Plotly figure
        """
        # Create dataframe
        df = pd.DataFrame({
            'UMAP1': umap_embeddings[:, 0],
            'UMAP2': umap_embeddings[:, 1],
            'Cluster': cluster_labels,
            'Text': texts,
            'Text_Length': [len(text.split()) for text in texts],
            'Text_Preview': [text[:50] + '...' if len(text) > 50 else text for text in texts]
        })
        
        # Create figure
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='Cluster',
            size='Text_Length',
            hover_data=['Text_Preview', 'Text_Length'],
            title='Interactive Cluster Visualization',
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='UMAP Component 1',
            yaxis_title='UMAP Component 2',
            legend_title='Cluster',
            width=900,
            height=700
        )
        
        # Update traces
        fig.update_traces(
            marker=dict(opacity=0.7),
            selector=dict(mode='markers')
        )
        
        return fig 