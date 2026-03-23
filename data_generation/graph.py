import json
from collections import deque  # For BFS traversal

class GraphNode:
    def __init__(self, entity_id, name):
        self.entity_id = entity_id  # Store the entity ID
        self.name = name
        self.incoming_edges = []  # List of tuples (source_node, edge_label)
        self.outgoing_edges = []  # List of tuples (target_node, edge_label)

    def add_edge(self, target, edge_label):
        """
        Adds a directed edge from this node to the target node.
        """
        self.outgoing_edges.append((target, edge_label))
        target.incoming_edges.append((self, edge_label))

    def remove_edge(self, target):
        """
        Removes the directed edge from this node to the target node.
        """
        self.outgoing_edges = [edge for edge in self.outgoing_edges if edge[0].entity_id != target.entity_id]
        target.incoming_edges = [edge for edge in target.incoming_edges if edge[0].entity_id != self.entity_id]

    @staticmethod
    def synchronize_graph_with_filtered_triples(root, removed_triples):
        """
        Traverses the graph starting from the root and removes edges that match the removed triples.

        Args:
            root (GraphNode): The root node of the graph.
            removed_triples (list): A list of triples to remove, in the format [(subject, predicate, object)].
        """
        # Convert removed_triples to a set for faster lookup
        removed_triples_set = set(removed_triples)

        # Use BFS to traverse the graph
        visited = set()
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            # Check outgoing edges
            for target, edge_label in list(node.outgoing_edges):  # Use list() to avoid modifying while iterating
                edge_triple = (node.name, edge_label, target.name)
                if edge_triple in removed_triples_set:
                    node.remove_edge(target)  # Remove the edge

            # Check incoming edges
            for source, edge_label in list(node.incoming_edges):  # Use list() to avoid modifying while iterating
                edge_triple = (source.name, edge_label, node.name)
                if edge_triple in removed_triples_set:
                    source.remove_edge(node)  # Remove the edge

            # Add neighbors to the queue
            for target, _ in node.outgoing_edges:
                if target not in visited:
                    queue.append(target)
            for source, _ in node.incoming_edges:
                if source not in visited:
                    queue.append(source)
    
    def get_path(self):
        """
        Returns all paths from the root(s) to this node as a list of paths.
        Each path is a list of triples in the format [source, edge_label, target].
        Uses iterative DFS to avoid recursion depth issues and handle cycles.
        """
        if not self.incoming_edges:
            return []  # This node is a root

        paths = []
        stack = [(self, [])]  # (node, current_path)
        visited = set()

        while stack:
            node, current_path = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            if not node.incoming_edges:
                # Reached a root node, add the completed path
                paths.append(current_path[::-1])  # Reverse to show root-to-leaf
            else:
                for source, edge_label in node.incoming_edges:
                    new_path = current_path + [[source.name, edge_label, node.name]]
                    stack.append((source, new_path))

        return paths

    def get_leaves(self):
        """
        Returns a list of all leaf nodes (nodes with no outgoing edges) as GraphNode objects.
        """
        leaves = []
        stack = [self]
        visited = set()
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            
            if not node.outgoing_edges:
                leaves.append(node)
            else:
                stack.extend([target for target, _ in node.outgoing_edges])
        return leaves

    def __repr__(self):
        """
        Defines the string representation of the GraphNode object.
        """
        return f"GraphNode(entity_id={self.entity_id}, name='{self.name}')"

    @staticmethod
    def extract_triplets_from_graph(root):
        """
        Extract all triples from the graph structure using BFS traversal.
        Traverses both outgoing and incoming edges to ensure all triples are included.
        Uses a set to avoid duplicate triples.
        
        Args:
            root (GraphNode): The root node of the graph.
        
        Returns:
            list: A list of triples in the format [(subject, predicate, object)].
        """
        triplets = []  # List to store extracted triples
        visited = set()  # Set to track visited nodes
        processed_edges = set()  # Set to track processed edges
        queue = deque([root])  # Queue for BFS traversal
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            
            # Add all outgoing edges as triples
            for target, edge_label in node.outgoing_edges:
                edge_triple = (node.name, edge_label, target.name)
                if edge_triple not in processed_edges:
                    triplets.append(edge_triple)
                    processed_edges.add(edge_triple)
                if target not in visited:
                    queue.append(target)
            
            # Add all incoming edges as triples
            for source, edge_label in node.incoming_edges:
                edge_triple = (source.name, edge_label, node.name)
                if edge_triple not in processed_edges:
                    triplets.append(edge_triple)
                    processed_edges.add(edge_triple)
                if source not in visited:
                    queue.append(source)
        
        return triplets
    
    def save_graph(self, filename):
        """
        Saves the graph structure starting from this node to a JSON file.
        Explicitly marks this node as the root.
        """
        graph_data = {
            'root': self._serialize()
        }
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def _serialize(self, visited=None):
        """
        Helper method to serialize the graph starting from this node.
        """
        if visited is None:
            visited = set()  # Track visited nodes by entity_id

        # If this node has already been serialized, return its basic info
        if self.entity_id in visited:
            return {
                'entity_id': self.entity_id,
                'name': self.name,
                'outgoing_edges': [],
                'incoming_edges': []
            }

        # Mark this node as visited
        visited.add(self.entity_id)

        # Serialize the current node
        serialized = {
            'entity_id': self.entity_id,
            'name': self.name,
            'outgoing_edges': [],
            'incoming_edges': []
        }

        # Serialize outgoing_edges
        for target, edge_label in self.outgoing_edges:
            serialized['outgoing_edges'].append({
                'target': target._serialize(visited),  # Pass visited set to avoid cycles
                'edge_label': edge_label
            })

        # Serialize incoming_edges
        for source, edge_label in self.incoming_edges:
            serialized['incoming_edges'].append({
                'source_id': source.entity_id,
                'source_name': source.name,
                'edge_label': edge_label
            })

        return serialized

    @staticmethod
    def load_graph(filename):
        """
        Loads the graph structure from a JSON file and returns the root node.
        """
        with open(filename, 'r') as f:
            graph_data = json.load(f)

        # First pass: Deserialize all nodes
        node_map = {}
        root = GraphNode._deserialize(graph_data['root'], node_map)

        # Second pass: Reconstruct incoming_edges
        GraphNode._reconstruct_incoming_edges(graph_data['root'], node_map)

        return root

    @staticmethod
    def _deserialize(data, node_map=None):
        """
        Helper method to deserialize the graph data and reconstruct the graph.
        """
        if node_map is None:
            node_map = {}  # Map entity_id to GraphNode objects

        # Create or retrieve the node
        if data['entity_id'] in node_map:
            node = node_map[data['entity_id']]
        else:
            node = GraphNode(data['entity_id'], data['name'])
            node_map[data['entity_id']] = node

        # Deserialize outgoing_edges
        for edge_data in data['outgoing_edges']:
            target_node = GraphNode._deserialize(edge_data['target'], node_map)
            node.add_edge(target_node, edge_data['edge_label'])

        return node

    @staticmethod
    def _reconstruct_incoming_edges(data, node_map):
        """
        Reconstruct incoming_edges after all nodes have been deserialized.
        """
        node = node_map[data['entity_id']]
        for edge_data in data['incoming_edges']:
            source_id = edge_data['source_id']
            if source_id in node_map:
                source_node = node_map[source_id]
                node.incoming_edges.append((source_node, edge_data['edge_label']))
            else:
                raise ValueError(f"Source node with entity_id={source_id} not found during deserialization.")