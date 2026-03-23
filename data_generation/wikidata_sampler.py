from rdflib import Graph, URIRef
from rdflib.namespace import RDFS, SKOS, DCTERMS
import random
import argparse
import time
import requests

# Define Wikidata API endpoint
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

def load_graph(file_path):
    """Load RDF file in appropriate format"""
    print(f"Loading graph from {file_path}...")
    start_time = time.time()
    g = Graph()
    
    # Determine format based on file extension
    if file_path.endswith('.nt'):
        format = 'ntriples'
    elif file_path.endswith('.rdf') or file_path.endswith('.xml'):
        format = 'xml'
    else:
        raise ValueError("Unsupported file format")
    
    g.parse(file_path, format=format)
    print(f"Loaded {len(g):,} triples in {time.time()-start_time:.1f}s")
    return g

def get_random_subject(graph):
    """Get random unique subject from the graph"""
    subjects = list(set(graph.subjects()))
    if not subjects:
        raise ValueError("No subjects found in the graph")
    return random.choice(subjects)

def get_label_from_wikidata(uri):
    """Fetch label for a URI from Wikidata"""
    try:
        # Extract QID from URI (e.g., http://www.wikidata.org/entity/Q12136 -> Q12136)
        qid = uri.split("/")[-1]
        if not qid.startswith("Q"):
            return None
        
        # Query Wikidata API
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json",
            "props": "labels",
            "languages": "en"
        }
        response = requests.get(WIKIDATA_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract English label
        if "entities" in data and qid in data["entities"]:
            labels = data["entities"][qid].get("labels", {})
            if "en" in labels:
                return labels["en"]["value"]
    except Exception as e:
        print(f"Error fetching label from Wikidata for {uri}: {str(e)}")
    return None

def get_node_info(graph, node):
    """Get readable information about a node"""
    # Check multiple label properties
    label_properties = [RDFS.label, SKOS.prefLabel, DCTERMS.title]
    name = None
    for prop in label_properties:
        name = graph.value(node, prop)
        if name:
            break
    
    # If no label found, try fetching from Wikidata
    if not name and str(node).startswith("http://www.wikidata.org/entity/"):
        name = get_label_from_wikidata(str(node))
    
    return {
        "uri": str(node),
        "name": str(name) if name else node.split("/")[-1],  # Fallback to URI fragment
        "predicates_count": len(list(graph.predicates(node)))
    }

def get_neighbors(graph, node):
    """Find and label neighboring nodes with both name and URI"""
    neighbors = []
    # Outgoing connections
    for s, p, o in graph.triples((node, None, None)):
        if isinstance(o, URIRef):
            name = None
            for prop in [RDFS.label, SKOS.prefLabel, DCTERMS.title]:
                name = graph.value(o, prop)
                if name:
                    break
            # If no label found, try fetching from Wikidata
            if not name and str(o).startswith("http://www.wikidata.org/entity/"):
                name = get_label_from_wikidata(str(o))
            neighbors.append({
                'uri': str(o),
                'name': str(name) if name else o.split("/")[-1],  # Fallback to URI fragment
                'direction': 'outgoing'
            })
    # Incoming connections
    for s, p, o in graph.triples((None, None, node)):
        if isinstance(s, URIRef):
            name = None
            for prop in [RDFS.label, SKOS.prefLabel, DCTERMS.title]:
                name = graph.value(s, prop)
                if name:
                    break
            # If no label found, try fetching from Wikidata
            if not name and str(s).startswith("http://www.wikidata.org/entity/"):
                name = get_label_from_wikidata(str(s))
            neighbors.append({
                'uri': str(s),
                'name': str(name) if name else s.split("/")[-1],  # Fallback to URI fragment
                'direction': 'incoming'
            })
    return neighbors

def format_neighbor_info(neighbor):
    """Format neighbor information for display"""
    direction_symbol = "→" if neighbor['direction'] == 'outgoing' else "←"
    return f"{direction_symbol} {neighbor['name']} <{neighbor['uri']}>"

def main():
    parser = argparse.ArgumentParser(description="Explore Wikidata subset")
    parser.add_argument("file", help="Path to downloaded RDF file")
    args = parser.parse_args()

    try:
        graph = load_graph(args.file)
        node = get_random_subject(graph)
        info = get_node_info(graph, node)
        neighbors = get_neighbors(graph, node)
        
        print("\n" + "="*50)
        print(f"Random Entity URI: {info['uri']}")
        print(f"Display Name: {info['name']}")
        print(f"Predicates Count: {info['predicates_count']}")
        print("\nNeighbors:")
        if neighbors:
            for neighbor in neighbors:
                print(format_neighbor_info(neighbor))
        else:
            print("No neighbors found")
        print("="*50)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()