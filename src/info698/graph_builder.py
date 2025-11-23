import networkx as nx
import numpy as np
from urllib.parse import urlparse
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict
from .embedding import CustomEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def extract_id(openalex_url):
    """Extract the ID from an OpenAlex URL (e.g., W3159481202 from https://openalex.org/W3159481202)."""
    return urlparse(openalex_url).path.split('/')[-1]


def build_citation_graph_v0(root_id, data, root_title=None):
    """
    Build a directed citation graph with edges from cited papers to the root paper. 
    This is the very initial implementation of the citatoion graph.
    """
    G = nx.DiGraph()
    root_id = extract_id(root_id)

    # Step 1: Add root paper node
    root_label = root_title if root_title else f"Paper {root_id}"
    G.add_node(root_id, label=root_label, type='root')

    # Step 2: Add cited papers as nodes
    cited_papers = set(data.keys())
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        title = data[pid].get('title', f"Paper {pid_extracted}")
        G.add_node(pid_extracted, label=title, type='cited')

    # Step 3: Add edges (from cited papers to root paper)
    edges = []
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        if pid_extracted != root_id:  # Avoid self-loop if root is in cited papers
            G.add_edge(pid_extracted, root_id)
            edges.append((pid_extracted, root_id))

    print(f"DEBUG: Generated {len(edges)} edges")  # Debugging output
    return G, edges


def build_citation_graph(root_id, data, root_title=None, embedding_model=None):
    """
    Build a directed citation graph with edges from cited papers to root and related works to cited papers.
    Optionally pre-compute node embeddings.
    
    Args:
        root_id: OpenAlex ID of the root paper.
        data: Citation data from OpenAlex (from citations.json).
        root_title: Title of the root paper (optional).
        embedding_model: SentenceTransformer model for embedding node labels (optional).
    
    Returns:
        G: NetworkX DiGraph.
        edges: List of edge tuples.
    """
    G = nx.DiGraph()
    root_id = extract_id(root_id)

    # Step 1: Add root paper node
    root_label = root_title if root_title else f"Paper {root_id}"
    root_attrs = {'label': root_label, 'type': 'root', 'cited_by_count': data.get(root_id, {}).get('cited_by_count', 0)}
    if embedding_model:
        # CustomEmbeddings.embed_query already returns a list
        root_attrs['embedding'] = embedding_model.embed_query(root_label)
    G.add_node(root_id, **root_attrs)

    # Step 2: Add cited papers as nodes
    cited_papers = set(data.keys())
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        title = data[pid].get('title', f"Paper {pid_extracted}")
        abstract = data[pid].get('abstract')
        attrs = {
            'label': title,
            'type': 'cited',
            'cited_by_count': data[pid].get('cited_by_count', 0),
            'publication_year': data[pid].get('publication_year', None),
            'authors': data[pid].get('authors', []),
            'venue': data[pid].get('venue', ''),
            'doi': data[pid].get('doi', None),
            'concepts': data[pid].get('concepts', []),
        }
        if embedding_model:
            # Prefer richer text when available
            text_for_embedding = f"{title}. {abstract}" if abstract else title
            attrs['embedding'] = embedding_model.embed_query(text_for_embedding)
        G.add_node(pid_extracted, **attrs)

    # Step 3: Add related works as nodes
    related_works = set()
    for pid in cited_papers:
        for rw in data[pid].get('related_works', []):
            rw_id = extract_id(rw)
            if rw_id not in cited_papers and rw_id != root_id:
                related_works.add(rw_id)
                attrs = {'label': f"Ref {rw_id}", 'type': 'related', 'cited_by_count': 0}
                if embedding_model:
                    # CustomEmbeddings.embed_query already returns a list
                    attrs['embedding'] = embedding_model.embed_query(attrs['label'])
                G.add_node(rw_id, **attrs)

    # Step 4: Add edges with weights
    edges = []
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        if pid_extracted != root_id:
            weight = data[pid].get('cited_by_count', 1) / 100.0  # Normalize weight
            G.add_edge(pid_extracted, root_id, weight=weight, etype='cited_root')
            edges.append((pid_extracted, root_id))
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        for rw in data[pid].get('related_works', []):
            rw_id = extract_id(rw)
            if rw_id in related_works:
                G.add_edge(rw_id, pid_extracted, weight=1.0, etype='related')  # Default weight for related works
                edges.append((rw_id, pid_extracted))

    # Step 4b: Add reference-based nodes and edges (reference -> citing paper)
    ref_edges_count = 0
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        ref_list = data[pid].get('references', [])
        if not ref_list:
            continue
        for ref in ref_list:
            ref_id = extract_id(ref)
            if ref_id == pid_extracted:
                continue
            # Ensure referenced node exists
            if ref_id not in G:
                ref_attrs = {'label': f"Ref {ref_id}", 'type': 'reference', 'cited_by_count': 0}
                if embedding_model:
                    # Use label for embedding if no title is available
                    ref_attrs['embedding'] = embedding_model.embed_query(ref_attrs['label'])
                G.add_node(ref_id, **ref_attrs)

            # Compute edge weight: combine citation signal and embedding similarity if available
            citation_norm = data[pid].get('cited_by_count', 1) / 100.0
            sim = 0.0
            src_emb = G.nodes[ref_id].get('embedding')
            dst_emb = G.nodes[pid_extracted].get('embedding')
            if src_emb is not None and dst_emb is not None:
                try:
                    sim = float(cosine_similarity([np.array(src_emb)], [np.array(dst_emb)])[0][0])
                except Exception:
                    sim = 0.0
            weight = 0.5 * citation_norm + 0.5 * max(sim, 0.0)
            G.add_edge(ref_id, pid_extracted, weight=weight, etype='reference')
            edges.append((ref_id, pid_extracted))
            ref_edges_count += 1

    print(
        f"DEBUG: Generated {len(edges)} edges "
        f"(cited→root: {len(cited_papers) - (1 if root_id in [extract_id(p) for p in cited_papers] else 0)}, "
        f"related→cited: {sum(len(data[pid].get('related_works', [])) for pid in cited_papers)}, "
        f"reference→cited: {ref_edges_count})"
    )
    return G, edges

def find_influential_papers(G: nx.DiGraph, top_k: int = 10) -> List[Dict[str, Any]]:
    """Find most influential papers using multiple centrality measures."""
    influential = []
    
    # Get centrality scores
    pagerank = nx.pagerank(G, weight='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight')
    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    
    for node in G.nodes():
        node_data = G.nodes[node]
        score = (
            0.4 * pagerank.get(node, 0) +
            0.3 * betweenness.get(node, 0) +
            0.3 * eigenvector.get(node, 0)
        )
        
        influential.append({
            'id': node,
            'title': node_data.get('label', ''),
            'type': node_data.get('type', ''),
            'cited_by_count': node_data.get('cited_by_count', 0),
            'publication_year': node_data.get('publication_year'),
            'influence_score': score,
            'pagerank': pagerank.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'eigenvector': eigenvector.get(node, 0)
        })
    
    # Sort by influence score
    influential.sort(key=lambda x: x['influence_score'], reverse=True)
    return influential[:top_k]

def find_temporal_trends(G: nx.DiGraph) -> Dict[str, Any]:
    """Analyze temporal trends in the citation graph."""
    trends = {}
    
    # Group nodes by year
    year_groups = defaultdict(list)
    for node in G.nodes():
        year = G.nodes[node].get('publication_year')
        if year:
            year_groups[year].append(node)
    
    if not year_groups:
        return trends
    
    # Calculate metrics per year
    years = sorted(year_groups.keys())
    trends['years'] = years
    trends['papers_per_year'] = [len(year_groups[year]) for year in years]
    
    # Citation patterns over time
    citations_per_year = []
    for year in years:
        year_citations = 0
        for node in year_groups[year]:
            year_citations += G.nodes[node].get('cited_by_count', 0)
        citations_per_year.append(year_citations)
    
    trends['citations_per_year'] = citations_per_year
    
    # Network growth
    cumulative_nodes = []
    cumulative_edges = []
    temp_G = nx.DiGraph()
    
    for year in years:
        for node in year_groups[year]:
            temp_G.add_node(node, **G.nodes[node])
            # Add edges to existing nodes
            for neighbor in G.successors(node):
                if neighbor in temp_G:
                    temp_G.add_edge(node, neighbor)
            for neighbor in G.predecessors(node):
                if neighbor in temp_G:
                    temp_G.add_edge(neighbor, node)
        
        cumulative_nodes.append(temp_G.number_of_nodes())
        cumulative_edges.append(temp_G.number_of_edges())
    
    trends['cumulative_nodes'] = cumulative_nodes
    trends['cumulative_edges'] = cumulative_edges
    
    return trends

def enhanced_graph_retrieval(G: nx.DiGraph, query: str, embedding_model=None, 
                           top_k: int = 5, use_temporal: bool = True,
                           use_communities: bool = True) -> List[Dict[str, Any]]:

    if embedding_model is None:
        # Fallback to simple text matching
        results = []
        query_lower = query.lower()
        for node in G.nodes(data=True):
            node_id, node_data = node
            label = node_data.get('label', '')
            if query_lower in label.lower():
                results.append({
                    'id': node_id,
                    'label': label,
                    'type': node_data.get('type', ''),
                    'score': 1.0,
                    'cited_by_count': node_data.get('cited_by_count', 0),
                    'publication_year': node_data.get('publication_year')
                })
        return results[:top_k]
    
    # Use embedding-based similarity
    query_embedding = embedding_model.embed_query(query)
    results = []
    
    for node in G.nodes(data=True):
        node_id, node_data = node
        
        # Semantic similarity
        if 'embedding' in node_data:
            node_embedding = np.array(node_data['embedding'])
            similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        else:
            # Fallback to text matching
            label = node_data.get('label', '')
            similarity = 1.0 if query.lower() in label.lower() else 0.0
        
        # Get centrality scores
        pagerank = nx.pagerank(G, weight='weight').get(node_id, 0)
        betweenness = nx.betweenness_centrality(G, weight='weight').get(node_id, 0)
        
        # Temporal score
        temporal_score = 1.0
        if use_temporal:
            year = node_data.get('publication_year')
            if year:
                # Recent papers get higher scores
                temporal_score = min(1.0, (2024 - year) / 10.0)
        
        # Community diversity score
        community_score = 1.0
        if use_communities:
            try:
                import networkx.algorithms.community as nx_comm
                communities = list(nx_comm.greedy_modularity_communities(G))
                for i, community in enumerate(communities):
                    if node_id in community:
                        community_score = 1.0 + (i * 0.1)  # Slight bonus for community diversity
                        break
            except:
                pass
        
        # Combined score
        combined_score = (
            0.4 * similarity +
            0.2 * pagerank +
            0.2 * betweenness +
            0.1 * temporal_score +
            0.1 * community_score
        )
        
        results.append({
            'id': node_id,
            'label': node_data.get('label', ''),
            'type': node_data.get('type', ''),
            'score': combined_score,
            'semantic_score': similarity,
            'pagerank_score': pagerank,
            'betweenness_score': betweenness,
            'temporal_score': temporal_score,
            'community_score': community_score,
            'cited_by_count': node_data.get('cited_by_count', 0),
            'publication_year': node_data.get('publication_year')
        })
    
    # Sort by combined score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


class GraphRetrival:
    def __init__(self):
          # Initialize graph and citation data
        
        self.embedding_model = CustomEmbeddings()
        try:
            with open("./data/citations.json", "r") as _file:
                self.citation_data = json.load(_file)
            
            if self.citation_data and len(self.citation_data) > 0:
                root_id = list(self.citation_data.keys())[0]
                # Derive root title from citation data when available
                root_block = self.citation_data.get(root_id, {})
                root_title = None
                if isinstance(root_block, dict):
                    root_title = root_block.get('title') or root_block.get('root_title')
                self.G, _ = build_citation_graph(
                    root_id,
                    self.citation_data[root_id],
                    root_title=root_title,
                    embedding_model=self.embedding_model,
                )
                
            else:
                print("Warning: No citation data found, creating empty graph")
                import networkx as nx
                self.G = nx.DiGraph()
                self.pagerank_scores = {}
                self.betweenness_scores = {}
                self.communities = []
                self.temporal_scores = {}
                
        except Exception as e:
            print(f"Warning: Could not load citation data: {e}")
            import networkx as nx
            self.G = nx.DiGraph()
            self.pagerank_scores = {}
            self.betweenness_scores = {}
            self.communities = []
            self.temporal_scores = {}
        
    def advanced_graph_retrieval(self, query: str, top_k: int = 5, max_hops: int = 3) -> List[Dict]:
        """Enhanced graph retrieval with multiple scoring factors."""
        query_embedding = self.embedding_model.embed_query(query)
        node_scores = []
        
        for node in self.G.nodes(data=True):
            node_id, node_data = node
            
            # Semantic similarity
            if 'embedding' in node_data:
                node_embedding = np.array(node_data['embedding'])
                semantic_score = cosine_similarity([query_embedding], [node_embedding])[0][0]
            else:
                # Fallback to text matching
                label = node_data.get('label', '')
                semantic_score = 1.0 if query.lower() in label.lower() else 0.0
            
            # PageRank score
            pagerank_score = self.pagerank_scores.get(node_id, 0)
            
            # Betweenness centrality
            betweenness_score = self.betweenness_scores.get(node_id, 0)
            
            # Temporal score
            temporal_score = self.temporal_scores.get(node_id, 0.5)
            
            # Citation count score
            cited_by_count = node_data.get('cited_by_count', 0)
            citation_score = min(cited_by_count / 100, 1.0)
            
            # Community diversity (nodes in different communities get bonus)
            community_bonus = 0
            for i, community in enumerate(self.communities):
                if node_id in community:
                    community_bonus = 0.1 * (i + 1)  # Slight bonus for community diversity
                    break
            
            # Combined score with weights
            combined_score = (
                0.4 * semantic_score +
                0.2 * pagerank_score +
                0.15 * betweenness_score +
                0.15 * temporal_score +
                0.1 * citation_score +
                community_bonus
            )
            
            node_scores.append({
                'id': node_id,
                'label': node_data.get('label', ''),
                'type': node_data.get('type', ''),
                'score': combined_score,
                'semantic_score': semantic_score,
                'pagerank_score': pagerank_score,
                'betweenness_score': betweenness_score,
                'temporal_score': temporal_score,
                'citation_score': citation_score,
                'cited_by_count': cited_by_count,
                'publication_year': node_data.get('publication_year')
            })
        
        # Sort by combined score
        node_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Expand to neighbors for context
        expanded_results = []
        for result in node_scores[:top_k]:
            expanded_results.append(result)
            
            # Add connected papers for context
            neighbors = list(self.G.successors(result['id'])) + list(self.G.predecessors(result['id']))
            for neighbor_id in neighbors[:2]:  # Limit to 2 neighbors per result
                if neighbor_id not in [r['id'] for r in expanded_results]:
                    neighbor_data = self.G.nodes[neighbor_id]
                    expanded_results.append({
                        'id': neighbor_id,
                        'label': neighbor_data.get('label', ''),
                        'type': neighbor_data.get('type', ''),
                        'score': result['score'] * 0.7,  # Reduced score for neighbors
                        'semantic_score': 0,
                        'pagerank_score': self.pagerank_scores.get(neighbor_id, 0),
                        'betweenness_score': self.betweenness_scores.get(neighbor_id, 0),
                        'temporal_score': self.temporal_scores.get(neighbor_id, 0.5),
                        'citation_score': min(neighbor_data.get('cited_by_count', 0) / 100, 1.0),
                        'cited_by_count': neighbor_data.get('cited_by_count', 0),
                        'publication_year': neighbor_data.get('publication_year')
                    })
        
        return expanded_results[:top_k * 2]  # Return more results for better fusion

    def _query_personalized_paths(self, question: str, top_k: int = 3):
        """Compute query-aware citation paths using personalized PageRank and shortest paths.
        Returns a list of human-readable path strings with edge labels and weights.
        """

        G = getattr(self, 'G', None)
        if not G or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
            return []

        # Identify root node
        root_candidates = [n for n, d in G.nodes(data=True) if d.get('type') == 'root']
        root = root_candidates[0] if root_candidates else None
        if root is None:
            return []

        # Compute query embedding
        try:
            q_emb = np.array(self.embedding_model.embed_query(question), dtype=float)
            if q_emb.ndim != 1:
                q_emb = q_emb.flatten()
        except Exception:
            q_emb = None

        def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
            try:
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na == 0 or nb == 0:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))
            except Exception:
                return 0.0

        # Personalization by similarity
        if q_emb is not None:
            personalization = {}
            for n, d in G.nodes(data=True):
                emb = d.get('embedding')
                if emb is None:
                    personalization[n] = 0.0
                else:
                    try:
                        personalization[n] = max(0.0, safe_cos(q_emb, np.array(emb, dtype=float)))
                    except Exception:
                        personalization[n] = 0.0
        else:
            personalization = {n: 1.0 for n in G.nodes}

        # Personalized PageRank
        try:
            pr = nx.pagerank(G, weight='weight', personalization=personalization)
        except Exception:
            pr = {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes}

        # Select anchors (exclude root)
        sorted_nodes = [n for n, _ in sorted(pr.items(), key=lambda kv: kv[1], reverse=True)]
        anchors = [n for n in sorted_nodes if n != root][: min(10, len(sorted_nodes))]

        # Shortest paths to root using inverse-weight cost
        def edge_cost(u, v, e):
            w = e.get('weight', 1.0)
            return 1.0 / (w + 1e-6)

        paths = []
        for a in anchors:
            try:
                p = nx.shortest_path(G, source=a, target=root, weight=lambda u, v, e: edge_cost(u, v, e))
                paths.append(p)
            except Exception:
                continue

        # Deduplicate and score
        seen = set()
        unique_paths = []
        for p in paths:
            key = tuple(p)
            if key not in seen:
                unique_paths.append(p)
                seen.add(key)

        def path_score(p):
            return sum(pr.get(n, 0.0) for n in p)

        unique_paths.sort(key=path_score, reverse=True)
        unique_paths = unique_paths[:top_k]

        # Render readable paths
        rendered = []
        for p in unique_paths:
            hops = []
            for i, n in enumerate(p):
                nd = G.nodes[n]
                title = nd.get('label', str(n))
                year = nd.get('publication_year')
                cited = nd.get('cited_by_count')
                parts = [title]
                suffix = []
                if isinstance(year, (int, float)):
                    suffix.append(str(int(year)))
                if isinstance(cited, (int, float)):
                    suffix.append(f"citations:{int(cited)}")
                if suffix:
                    parts.append(f"({', '.join(suffix)})")
                hops.append(" ".join(parts))

                if i < len(p) - 1:
                    u, v = p[i], p[i+1]
                    ed = G.get_edge_data(u, v) or {}
                    etype = ed.get('etype', 'edge')
                    w = ed.get('weight', 0.0)
                    hops.append(f" —{etype} (w={w:.2f})→ ")
            rendered.append("".join(hops))

        return rendered


def visualize_static(G, edges):
    """Visualize the graph using Matplotlib with visible edges."""
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    plt.figure(figsize=(12, 8))
    node_colors = ['red' if node_types[n] == 'root' else 'lightblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color=node_colors,
            font_size=8, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20)
    plt.title("Static Citation Graph")
    plt.show()

def visualize_interactive(G, edges, root_id):
    """Visualize the graph interactively using Plotly with visible edges."""
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} → {edge[1]}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#555'),
        hoverinfo='text',
        text=edge_text[::3],
        mode='lines'
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_hover_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels[node])
        nd = G.nodes[node]
        t = nd.get('type', node_types.get(node, ''))
        yr = nd.get('publication_year')
        cit = nd.get('cited_by_count')
        authors = nd.get('authors') or []
        venue = nd.get('venue') or ''
        doi = nd.get('doi') or ''
        concepts = nd.get('concepts') or []
        openalex_id = nd.get('openalex_id') or ''
        oa_url = nd.get('oa_url') or ''
        parts = [str(labels.get(node, node))]
        meta = []
        if t:
            meta.append(f"Type: {t}")
        if isinstance(yr, (int, float)):
            meta.append(f"Year: {int(yr)}")
        if isinstance(cit, (int, float)):
            meta.append(f"Citations: {int(cit)}")
        # Optional enriched fields (keep concise)
        if isinstance(authors, list) and authors:
            a = ", ".join([str(a) for a in authors[:5] if a])
            if a:
                meta.append(f"Authors: {a}{'…' if len(authors) > 5 else ''}")
        if isinstance(venue, str) and venue:
            meta.append(f"Venue: {venue}")
        if isinstance(doi, str) and doi:
            meta.append(f"DOI: {doi}")
        if isinstance(concepts, list) and concepts:
            c = ", ".join([str(c) for c in concepts[:3] if c])
            if c:
                meta.append(f"Concepts: {c}{'…' if len(concepts) > 3 else ''}")
        if isinstance(openalex_id, str) and openalex_id:
            meta.append(f"OpenAlex: {openalex_id}")
        if isinstance(oa_url, str) and oa_url:
            meta.append(f"OA: {oa_url}")
        if meta:
            parts.append("<br>" + " | ".join(meta))
        node_hover_text.append("".join(parts))
        node_colors.append('red' if node == root_id else 'lightblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        hovertext=node_hover_text,
        marker=dict(size=20, color=node_colors, line=dict(width=2))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Citation Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # fig.show()
    return fig

def main(root_id, data, root_title=None):
    """Main function to build and visualize the citation graph."""
    G, edges = build_citation_graph(root_id, data, root_title)
    
    # Print nodes and edges
    print("Nodes:")
    for node, attr in G.nodes(data=True):
        print(f"- {node}: {attr['label']} ({attr['type']})")
    print("\nEdges:")
    if not edges:
        print("No edges found in the graph.")
    for citing, cited in edges:
        print(f"- {citing} → {cited}")
    
    # Visualize the graph
    # visualize_static(G, edges)
    visualize_interactive(G, edges, extract_id(root_id))

def analyze_graph_structure(G: nx.DiGraph) -> Dict[str, Any]:
    """Analyze graph structure and compute advanced metrics."""
    analysis = {}
    
    # Basic metrics
    analysis['nodes'] = G.number_of_nodes()
    analysis['edges'] = G.number_of_edges()
    analysis['density'] = nx.density(G)
    analysis['is_connected'] = nx.is_weakly_connected(G)
    
    # Centrality measures
    analysis['pagerank'] = nx.pagerank(G, weight='weight')
    analysis['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    analysis['closeness'] = nx.closeness_centrality(G)
    analysis['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    
    # Community detection
    try:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G))
        analysis['communities'] = communities
        analysis['modularity'] = nx_comm.modularity(G, communities)
    except:
        analysis['communities'] = []
        analysis['modularity'] = 0
    
    # Temporal analysis
    years = [G.nodes[node].get('publication_year') for node in G.nodes() 
            if G.nodes[node].get('publication_year')]
    if years:
        analysis['year_range'] = (min(years), max(years))
        analysis['avg_year'] = np.mean(years)
        analysis['year_std'] = np.std(years)
    
    # Citation patterns
    citation_counts = [G.nodes[node].get('cited_by_count', 0) for node in G.nodes()]
    if citation_counts:
        analysis['avg_citations'] = np.mean(citation_counts)
        analysis['max_citations'] = max(citation_counts)
        analysis['citation_std'] = np.std(citation_counts)
    
    return analysis


if __name__ == "__main__":
    # paper = "Attention is all you need"
    # openalex_api = OpenAlexAPI(paper)
    # data = openalex_api.get_citations()
    import simplejson as json
    with open("./data/citations.json", "r") as _file:
        data = json.load(_file)
    # main(
    #     root_id=openalex_api.query,
    #     data=data[openalex_api.query_alex_repsone.get('id', "root")],
    #     root_title=paper
    # )

    main(
        root_id=list(data.keys())[0],
        data=data[list(data.keys())[0]],
        root_title="Attention is all you need"
    )