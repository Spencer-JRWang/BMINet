import networkx as nx
import numpy as np
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import community as community_louvain
import os

class NetworkVisualizer:
    def __init__(self, graph, metrics):
        """
        Initialize the NetworkVisualizer with a graph, metrics, and an output directory.

        :param graph: A NetworkX graph object to visualize.
        :param metrics: A dictionary with computed metrics.
        :param output_dir: Directory to save output figures.
        """
        self.graph = graph
        self.metrics = metrics

    def plot_adjacency_matrix(self, pdf_path=None):
        """
        Plot and optionally save the adjacency matrix heatmap of the graph.

        :param save_pdf: Boolean to decide whether to save the plot as PDF.
        :param pdf_path: Path to save the PDF file if save_pdf is True.
        """
        adj_matrix = self.metrics['adj_matrix']

        plt.figure(figsize=(10, 10))
        sns.heatmap(adj_matrix, cmap=sns.color_palette(['white', '#1f77b4']), annot=False, cbar=False, square=True,
                    xticklabels=self.graph.nodes(), yticklabels=self.graph.nodes(), linewidths=0.7, linecolor='gray')
        plt.title('Adjacency Matrix Heatmap')
        plt.tight_layout()
        plt.show()

        if pdf_path:
            pdf_path = os.path.join(pdf_path, 'Adj.pdf')
            plt.savefig(pdf_path)
            print(f'Adjacency matrix heatmap saved to {pdf_path}')
        plt.close()

    def plot_centrality_measures(self, pdf_path=None):
        """
        Plot and optionally save the centrality measures including betweenness, closeness, degree, and clustering coefficient.

        :param save_pdf: Boolean to decide whether to save the plot as PDF.
        :param pdf_path: Path to save the PDF file if save_pdf is True.
        """
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.bar(self.metrics['betweenness_sorted'].keys(), self.metrics['betweenness_sorted'].values(), color='skyblue')
        plt.title('Betweenness Centrality')
        plt.xticks(rotation=90)

        plt.subplot(2, 2, 2)
        plt.bar(self.metrics['closeness_sorted'].keys(), self.metrics['closeness_sorted'].values(), color='lightgreen')
        plt.title('Closeness Centrality')
        plt.xticks(rotation=90)

        plt.subplot(2, 2, 3)
        plt.bar(self.metrics['degree_sorted'].keys(), self.metrics['degree_sorted'].values(), color='salmon')
        plt.title('Degree')
        plt.xticks(rotation=90)

        plt.subplot(2, 2, 4)
        plt.bar(self.metrics['clustering_sorted'].keys(), self.metrics['clustering_sorted'].values(), color='#FABB6E')
        plt.title('Clustering Coefficient')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.title('Centrality Distribution Plot')
        plt.show()

        if pdf_path:
            pdf_path = os.path.join(pdf_path, 'Centrality.pdf')
            plt.savefig(pdf_path)
            print(f'Centrality distribution plot saved to {pdf_path}')
        plt.close()

    def plot_network_communities(self, pdf_path=None):
        """
        Plot and optionally save the network with community clustering.

        :param save_pdf: Boolean to decide whether to save the plot as PDF.
        :param pdf_path_template: Template path to save the PDF files if save_pdf is True.
        """
        
        partition = community_louvain.best_partition(self.graph)
        communities = set(partition.values())
        num_communities = len(communities)
        cmap = plt.get_cmap('Set2', num_communities)
        pos = nx.spring_layout(self.graph, seed=42)

        plt.figure(figsize=(12, 12))
        colors = [cmap(partition[node]) for node in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color=colors, edgecolors='black')
        nx.draw_networkx_edges(self.graph, pos, edgelist=nx.edges(self.graph), edge_color='gray', width=2)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        plt.title('Network Clustering Visualization with Separated Communities')
        plt.show()

        if pdf_path:
            pdf_path = os.path.join(pdf_path, 'Communities.pdf')
            plt.savefig(pdf_path)
            print(f'Network clustering visualization saved to {pdf_path}')
        plt.close()


class NetworkMetrics:
    def __init__(self, graph):
        """
        Initialize the NetworkMetrics with a graph.

        :param graph: A NetworkX graph object to analyze.
        """
        self.graph = graph

    def compute_metrics(self):
        """
        Compute various network metrics including betweenness, closeness, degree, and clustering coefficient.

        :return: A dictionary with computed metrics.
        """
        metrics = {}
        metrics['betweenness'] = nx.betweenness_centrality(self.graph)
        metrics['closeness'] = nx.closeness_centrality(self.graph)
        metrics['degree'] = dict(self.graph.degree())
        metrics['clustering'] = nx.clustering(self.graph)

        # Sort metrics
        metrics['betweenness_sorted'] = dict(sorted(metrics['betweenness'].items(), key=lambda item: item[1], reverse=True))
        metrics['closeness_sorted'] = dict(sorted(metrics['closeness'].items(), key=lambda item: item[1], reverse=True))
        metrics['degree_sorted'] = dict(sorted(metrics['degree'].items(), key=lambda item: item[1], reverse=True))
        metrics['clustering_sorted'] = dict(sorted(metrics['clustering'].items(), key=lambda item: item[1], reverse=True))
        adj_matrix = nx.to_numpy_array(self.graph)
        adj_matrix = np.where(adj_matrix > 0, 1, 0)
        metrics['adj_matrix'] = adj_matrix

        return metrics
