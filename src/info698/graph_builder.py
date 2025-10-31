import networkx as nx
from urllib.parse import urlparse
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def extract_id(openalex_url):
    """Extract the ID from an OpenAlex URL (e.g., W3159481202 from https://openalex.org/W3159481202)."""
    return urlparse(openalex_url).path.split('/')[-1]


def build_citation_graph(root_id, data, root_title=None):
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
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels[node])
        node_colors.append('red' if node == root_id else 'lightblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
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

    fig.show()
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