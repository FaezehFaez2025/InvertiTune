import sys
import os
import random
import math
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
import tkinter as tk
from tqdm import tqdm
from tkinter import messagebox, simpledialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from graph import GraphNode
from rule_based_triple_filtering import filter_triple

# Entity IDs to exclude from expansion in controlled mode
EXPANSION_BLACKLIST = {
    "Q6581097",  # male
    "Q5",        # human
    "Q12308941", # male given name
    "Q51929218", # first-person singular
    "Q51929403", # second-person plural
    "Q6581072",  # female
    "Q618779",   # award
    "Q28640",    # profession
    "Q12047083", # professionalism
    "Q19652",    # public domain
    "Q101352",   # family name
    "Q113159385",# right-handed person
    "Q2421902",  # handedness
    "Q789447",   # left-handedness
    "Q3039938",  # right-handedness
    "Q73555012", # works protected by copyrights
    "Q1860",     # English
    "Q8229",     # Latin script
    "Q4220917",  # film award
    "Q71887839", # copyrights on works have expired
    "Q3739104",  # natural causes
    "Q82955",    # politician
    "Q11879590", # female given name
    "Q84048852", # female human
    "Q467",      # woman
    "Q3031",     # girl
    "Q188830",   # wife
    "Q1196129",  # spouse
    "Q28747937", # history of a city
    "Q3331189",  # version, edition or translation
    "Q4663903",  # Wikimedia portal
    "Q4164871",  # position
    "Q192581",   # job activity
    "Q268378",   # work
    "Q486972",   # human settlement
    "Q32022732", # Portal:Human settlements
    "Q203516",   # birth rate
    "Q10815002", # Portal:Family
    "Q8436",     # family
    "Q13780930", # worldwide
    "Q14565199", # right
    "Q542952",   # left and right
    "Q13196750", # left
    "Q10764194", # minus sign
    "Q16695773", # WikiProject
    "Q24025284", # sometimes changes
    "Q26256810", # topic
    "Q2366457",  # department
    "Q17172850", # voice
    "Q3348297"   # observer
}

PREDICATE_BLACKLIST = {
    "different from",
    "subclass of",
}

# Endpoint URLs
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
YAGO_ENDPOINT = "https://yago-knowledge.org/sparql/query"

def get_entity_name(entity_id, source):
    if source == "wikidata":
        sparql_query_src_entity_label = \
        """SELECT ?srcLabel
        WHERE 
        {
          BIND(wd:%s AS ?src) .
          ?src rdfs:label ?srcLabel .
          FILTER(LANG(?srcLabel) = "en")
        }""" % entity_id
    elif source == "yago":
        sparql_query_src_entity_label = \
        """PREFIX yago: <http://yago-knowledge.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?srcLabel
        WHERE 
        {
          yago:%s rdfs:label ?srcLabel .
          FILTER(LANG(?srcLabel) = "en")
        }""" % entity_id

    try:
        endpoint_url = WIKIDATA_ENDPOINT if source == "wikidata" else YAGO_ENDPOINT
        r = get_results(endpoint_url, sparql_query_src_entity_label)['results']['bindings']
        if len(r) == 0:
            return None
        src_entity_name = r[0]['srcLabel']['value']
        return src_entity_name
    except Exception as e:
        print(f"Error getting entity name: {e}")
        return None

def query(entity_id, source):
    if source == "wikidata":
        sparql_query_relations = """
        SELECT ?realrelationLabel ?x ?xLabel
        WHERE 
        {
          wd:%s ?relation ?x .
          ?realrelation wikibase:directClaim ?relation
          SERVICE wikibase:label { 
            bd:serviceParam wikibase:language "en". 
          }  
        }""" % entity_id
    elif source == "yago":
        sparql_query_relations = """
        PREFIX yago: <http://yago-knowledge.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?relation ?neighbor ?neighborLabel
        WHERE 
        {
          yago:%s ?relation ?neighbor .
          ?neighbor rdfs:label ?neighborLabel .
          FILTER(LANG(?neighborLabel) = "en")
        }""" % entity_id

    try:
        endpoint_url = WIKIDATA_ENDPOINT if source == "wikidata" else YAGO_ENDPOINT
        results = get_results(endpoint_url, sparql_query_relations)
        src_entity_name = get_entity_name(entity_id, source)

        if src_entity_name is None:
            return [], []

        if source == "wikidata":
            tail_entities = [(result['x']['value'].split('/')[-1], result['xLabel']['value']) for result in results["results"]["bindings"] if 'http://www.wikidata.org/entity' in result['x']['value']]
            triplets = [(src_entity_name, result['realrelationLabel']['value'], result['xLabel']['value']) for result in results["results"]["bindings"]]
        elif source == "yago":
            tail_entities = [(result['neighbor']['value'].split('/')[-1], result['neighborLabel']['value']) for result in results["results"]["bindings"]]
            triplets = [(src_entity_name, result['relation']['value'].split('/')[-1], result['neighborLabel']['value']) for result in results["results"]["bindings"]]

        return tail_entities, triplets
    except Exception as e:
        print(f"Error querying relations: {e}")
        return [], []

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return {}

def get_n_hop_neighbors(entity_id, n, source, ratio=1.0):
    if n == 0:
        return set(), set()

    tail_entities, triplets = query(entity_id, source)
    all_tail_entities = set(tail_entities)
    all_triplets = set(triplets)

    if n > 1:
        # Calculate the number of neighbors to explore for the next hop
        num_neighbors_to_explore = math.ceil(ratio * len(tail_entities))
        
        # Randomly select a subset of neighbors to explore
        neighbors_to_explore = random.sample(tail_entities, num_neighbors_to_explore) if num_neighbors_to_explore < len(tail_entities) else tail_entities
        
        for tail_entity_id, _ in neighbors_to_explore:
            new_tail_entities, new_triplets = get_n_hop_neighbors(tail_entity_id, n-1, source, ratio)
            all_tail_entities.update(new_tail_entities)
            all_triplets.update(new_triplets)

    return all_tail_entities, all_triplets

def get_entity_list_query(source, type_qid=None):
    """Generate a SPARQL query to retrieve a list of entities based on the source."""
    if source == "wikidata":
        if type_qid:
            return f"""
            SELECT ?entity ?label WHERE {{
              ?entity wdt:P31 wd:{type_qid} ;
                     rdfs:label ?label.
              FILTER(LANG(?label) = "en")
            }}
            LIMIT 20000
            """
        else:
            return """
            SELECT ?entity ?label WHERE {
              ?entity rdfs:label ?label.
              FILTER(LANG(?label) = "en")
            }
            LIMIT 20000
            """
    elif source == "yago":
        return """
        PREFIX yago: <http://yago-knowledge.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?entity ?label WHERE {
          ?entity rdfs:label ?label.
          FILTER(LANG(?label) = "en")
        }
        LIMIT 20000
        """
    else:
        raise ValueError(f"Unsupported source: {source}")

def select_entity_from_yago():
    """Display a GUI menu to select an entity from YAGO."""
    # Query YAGO to retrieve a list of entities
    sparql_query = get_entity_list_query("yago")
    results = get_results(YAGO_ENDPOINT, sparql_query)['results']['bindings']

    if not results:
        messagebox.showerror("Error", "No entities found in YAGO.")
        return None

    # Create a GUI to display the entities
    root = tk.Tk()
    root.title("Select an Entity from YAGO")
    root.geometry("600x400")

    # Frame to hold the listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Add a label to show the total number of entities
    total_entities = len(results)
    label = tk.Label(frame, text=f"Total Entities: {total_entities}", font=("Arial", 12))
    label.pack(pady=5)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Add a listbox
    label_listbox = tk.Listbox(frame, font=("Arial", 12), yscrollcommand=scrollbar.set)
    label_listbox.pack(fill=tk.BOTH, expand=True)

    # Link the scrollbar to the listbox
    scrollbar.config(command=label_listbox.yview)

    # Populate the listbox with entities
    entities = []
    for result in results:
        entity_uri = result['entity']['value']
        entity_label = result['label']['value']
        label_listbox.insert(tk.END, f"{entity_label} ({entity_uri})")
        entities.append((entity_uri, entity_label))

    selected_entity = None

    def on_select():
        nonlocal selected_entity
        index = label_listbox.curselection()
        if index:
            selected_entity = entities[index[0]]
            root.destroy()

    select_button = tk.Button(root, text="Select", command=on_select, font=("Arial", 12))
    select_button.pack(pady=10)

    root.mainloop()

    return selected_entity
    
def select_entity_from_wikidata(type_qid=None):
    """Display a GUI menu to select an entity from Wikidata."""
    # Query Wikidata to retrieve a list of entities
    sparql_query = get_entity_list_query("wikidata", type_qid)
    results = get_results(WIKIDATA_ENDPOINT, sparql_query)['results']['bindings']

    if not results:
        messagebox.showerror("Error", "No entities found in Wikidata.")
        return None

    # Create a GUI to display the entities
    root = tk.Tk()
    root.title("Select an Entity from Wikidata")
    root.geometry("600x400")

    # Frame to hold the listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Add a label to show the total number of entities
    total_entities = len(results)
    label = tk.Label(frame, text=f"Total Entities: {total_entities}", font=("Arial", 12))
    label.pack(pady=5)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Add a listbox
    label_listbox = tk.Listbox(frame, font=("Arial", 12), yscrollcommand=scrollbar.set)
    label_listbox.pack(fill=tk.BOTH, expand=True)

    # Link the scrollbar to the listbox
    scrollbar.config(command=label_listbox.yview)

    # Populate the listbox with entities
    entities = []
    for result in results:
        entity_uri = result['entity']['value']
        entity_label = result['label']['value']
        label_listbox.insert(tk.END, f"{entity_label} ({entity_uri})")
        entities.append((entity_uri, entity_label))

    selected_entity = None

    def on_select():
        nonlocal selected_entity
        index = label_listbox.curselection()
        if index:
            selected_entity = entities[index[0]]
            root.destroy()

    select_button = tk.Button(root, text="Select", command=on_select, font=("Arial", 12))
    select_button.pack(pady=10)

    root.mainloop()

    return selected_entity

def create_gui(triplets):
    selected_triplets = []
    root = tk.Tk()
    root.title("Select Triples Using Checkboxes")
    root.geometry("800x600")

    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    # Create canvas with scrollbar
    canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Configure canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # macOS-specific touchpad scrolling
    def _on_macos_scroll(event):
        # Handle both touchpad and mouse wheel events
        if event.delta:
            # Calculate scroll amount based on delta
            scroll_amount = -int(event.delta)
            canvas.yview_scroll(scroll_amount, "units")
            # Update scrollbar position
            canvas.update_idletasks()
            scrollbar.set(*canvas.yview())

    # Bind macOS-specific scroll events
    canvas.bind_all("<MouseWheel>", _on_macos_scroll)

    # Configure scroll region
    def configure_scroll_region(event):
        bbox = canvas.bbox("all")
        canvas.configure(scrollregion=bbox)
        # Update scrollbar position after resizing
        canvas.update_idletasks()
        scrollbar.set(*canvas.yview())

    scrollable_frame.bind("<Configure>", configure_scroll_region)

    # Checkbox variables
    check_vars = []
    checkboxes = []

    # Create header
    header_frame = tk.Frame(scrollable_frame)
    header_frame.pack(fill=tk.X, pady=5)
    
    all_var = tk.BooleanVar()
    select_all_cb = tk.Checkbutton(
        header_frame,
        text="Select All/None",
        variable=all_var,
        command=lambda: toggle_all(all_var.get()),
        font=("Arial", 12, "bold")
    )
    select_all_cb.pack(side=tk.LEFT, padx=5)

    # Create triple checkboxes
    for idx, triplet in enumerate(triplets):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        var = tk.BooleanVar(value=False)
        check_vars.append(var)
        
        cb = tk.Checkbutton(
            frame,
            variable=var,
            command=lambda v=var, t=triplet: update_selection(v, t),
            text=f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]',
            font=("Arial", 11),
            wraplength=700,
            justify=tk.LEFT
        )
        cb.pack(side=tk.LEFT, anchor="w")
        checkboxes.append(cb)

    def toggle_all(select_all):
        for var in check_vars:
            var.set(select_all)
        update_selection_list()

    def update_selection(var, triplet):
        if var.get():
            if triplet not in selected_triplets:
                selected_triplets.append(triplet)
        else:
            if triplet in selected_triplets:
                selected_triplets.remove(triplet)
        # Update select all checkbox state
        all_selected = all(var.get() for var in check_vars)
        all_var.set(all_selected)

    def update_selection_list():
        selected_triplets.clear()
        for var, triplet in zip(check_vars, triplets):
            if var.get():
                selected_triplets.append(triplet)

    def save_selected():
        if not selected_triplets:
            messagebox.showinfo("Info", "No triples selected!")
            return

        # Create the data folder if it doesn't exist
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Create the source folder (wikidata or yago) if it doesn't exist
        source_folder = os.path.join(data_folder, args.source)
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)

        # Get the entity name for the file name
        entity_name = get_entity_name(args.entity, args.source)
        if not entity_name:
            entity_name = args.entity  # Fallback to entity ID if name is not available

        # Create the file name with entity ID, name, hops, and ratio
        file_name = f"{args.entity}_{entity_name}_{args.hops}_hop_{args.ratio}_ratio_triples.txt"
        file_path = os.path.join(source_folder, file_name)

        # Save the selected triples to the file
        with open(file_path, "w") as f:
            for triplet in selected_triplets:
                f.write(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]\n')

        messagebox.showinfo("Success", f"Saved {len(selected_triplets)} triples to {file_path}")
        root.destroy()

    # Save button
    save_btn = tk.Button(root, text="Save Selected", command=save_selected, font=("Arial", 12))
    save_btn.pack(pady=10)

    root.mainloop()

def sanitize_filename(name):
    # Replace invalid characters with an underscore
    sanitized_name = re.sub(r'[/]', '_', name)
    return sanitized_name

def process_single_sample(entity, hops, source, ratio):
    """Process a single sample and return the results."""
    entity_id = entity['entity']['value'].split('/')[-1]
    print(f"\nProcessing entity: {entity_id} with {hops} hops")

    tail_entities, triplets = get_n_hop_neighbors(entity_id, hops, source, ratio)
    triplets = list(triplets)  # Convert to list for consistent ordering

    print('\nTail Entity IDs: ', [e[0] for e in tail_entities])
    print('\nTail Entity Names: ', [e[1] for e in tail_entities])

    print("\nTriplets:")
    for triplet in triplets:
        print(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]')

    # Save the triples to a file
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    source_folder = os.path.join(data_folder, source)
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    entity_name = get_entity_name(entity_id, source)
    if not entity_name:
        entity_name = entity_id  # Fallback to entity ID if name is not available

    # Sanitize the entity name
    sanitized_entity_name = sanitize_filename(entity_name)

    # Create the file name with entity ID, sanitized name, hops, and ratio
    file_name = f"{entity_id}_{sanitized_entity_name}_{hops}_hop_{ratio}_ratio_triples.txt"
    file_path = os.path.join(source_folder, file_name)

    with open(file_path, "w") as f:
        for triplet in triplets:
            f.write(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]\n')

    print(f"Saved {len(triplets)} triples to {file_path}")
    return file_path

def process_single_sample_controlled(entity, hops, source, num_neighbors_per_hop):
    """
    Process a single sample using controlled extraction and return the results.
    """
    entity_id = entity['entity']['value'].split('/')[-1]
    print(f"\nProcessing entity: {entity_id} with {hops} hops (controlled extraction)")

    # Use controlled extraction
    root_node = get_controlled_neighbors(entity_id, hops, source, num_neighbors_per_hop)
    triplets = extract_triplets_from_graph(root_node)

    print('\nTail Entity Names: ', list(set([t[2] for t in triplets])))  # Unique tail names

    if not triplets:
        print(f"Skipping entity {entity_id}: empty subgraph, nothing to save.")
        return None

    print("\nTriplets:")
    for triplet in triplets:
        print(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]')

    # Save the triples to a file
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    source_folder = os.path.join(data_folder, source)
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    entity_name = get_entity_name(entity_id, source)
    if not entity_name:
        entity_name = entity_id  # Fallback to entity ID if name is not available

    # Sanitize the entity name
    sanitized_entity_name = sanitize_filename(entity_name)

    # Create the file name with entity ID, sanitized name, hops, and ratio
    file_name = f"{entity_id}_{sanitized_entity_name}_{hops}_hop_{num_neighbors_per_hop}_neighbors_triples.txt"
    file_path = os.path.join(source_folder, file_name)

    with open(file_path, "w") as f:
        for triplet in triplets:
            f.write(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]\n')

    print(f"Saved {len(triplets)} triples to {file_path}")

    # Save the graph as a .json file with the same name
    json_file_path = os.path.join(source_folder, file_name.replace(".txt", ".json"))
    root_node.save_graph(json_file_path)
    print(f"Saved graph structure to {json_file_path}")

    return file_path

def process_multiple_samples(num_samples, max_hops, source, ratio, parallel=False, num_threads=1, controlled_extraction=False, num_neighbors_per_hop=3, type_qid=None,
                             resume_generation=False):
    """Process multiple samples with optional parallel execution."""
    # Query the knowledge base to retrieve a list of entities
    sparql_query = get_entity_list_query(source, type_qid)
    if source == "wikidata":
        endpoint_url = WIKIDATA_ENDPOINT
    elif source == "yago":
        endpoint_url = YAGO_ENDPOINT

    results = get_results(endpoint_url, sparql_query)['results']['bindings']

    if not results:
        print("No entities found in the knowledge base.")
        return

    # If resume_generation mode is enabled, filter out already processed entities
    if resume_generation:
        source_folder = os.path.join("data", source)
        processed_entities = set()
        if os.path.exists(source_folder):
            for file in os.listdir(source_folder):
                if file.endswith("_triples.txt"):
                    # Assumes file name format: "<entity_id>_..."
                    entity_id = file.split('_')[0]
                    processed_entities.add(entity_id)
        original_count = len(results)
        results = [entity for entity in results
                   if entity['entity']['value'].split('/')[-1] not in processed_entities]
        filtered_count = len(results)
        print(f"Resume mode: Filtered out {original_count - filtered_count} processed entities. "
              f"{filtered_count} new entities remaining.")
        if filtered_count < num_samples:
            print(f"Warning: Only {filtered_count} new entities available. Adjusting number of samples to {filtered_count}.")
            num_samples = filtered_count

    # Randomly select entities from the remaining new entities
    selected_entities = random.sample(results, num_samples)

    saved_count = 0

    if parallel:
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for entity in selected_entities:
                hops = random.randint(max_hops, max_hops)
                if controlled_extraction:
                    futures.append(executor.submit(process_single_sample_controlled, entity, hops, source, num_neighbors_per_hop))
                else:
                    futures.append(executor.submit(process_single_sample, entity, hops, source, ratio))

            # Initialize progress bar
            with tqdm(total=num_samples, desc="Processing samples", unit="sample") as pbar:
                for future in as_completed(futures):
                    result = future.result()  # Wait for the task to complete
                    if result is not None:
                        saved_count += 1
                    pbar.update(1)
    else:
        # Sequential execution
        with tqdm(total=num_samples, desc="Processing samples", unit="sample") as pbar:
            for entity in selected_entities:
                hops = random.randint(max_hops, max_hops)
                if controlled_extraction:
                    result = process_single_sample_controlled(entity, hops, source, num_neighbors_per_hop)
                else:
                    result = process_single_sample(entity, hops, source, ratio)
                if result is not None:
                    saved_count += 1
                pbar.update(1)

    print(f"\nDone. Saved {saved_count} out of {num_samples} subgraphs (skipped {num_samples - saved_count} empty).")

def get_controlled_neighbors(entity_id, hops, source, num_neighbors_per_hop):
    """
    Controlled sampling of neighbors with rule-based validation.
    Retrieves a specified number of valid neighbors per node and builds a graph structure.
    Skips expansion for entities in the EXPANSION_BLACKLIST.
    Enforces predicate uniqueness per node. Discards triplets with repeated predicates (same node→predicate→different objects) as they produce unanswerable multi-hop questions (multiple valid paths exist, no single correct answer).
    Excludes blacklisted predicates.
    Special handling for "instance of" predicates: keeps the triple but doesn't expand the object entity.
    """
    # Get the name of the root entity
    root_name = get_entity_name(entity_id, source) or entity_id
    # Create the root node of the graph
    root = GraphNode(entity_id, root_name)
    # Dictionary to track all nodes in the graph
    all_nodes = {entity_id: root}
    # List of nodes to expand in the current hop
    nodes_to_expand = [root]

    # Iterate through each hop
    for current_hop in range(hops):
        next_nodes = []  # Nodes to expand in the next hop
        for current_node in nodes_to_expand:
            # Skip expansion if this entity is in the blacklist
            if current_node.entity_id in EXPANSION_BLACKLIST:
                print(f"Skipping expansion for blacklisted entity: {current_node.entity_id} ({current_node.name})")
                continue

            # Query the current node to get its neighbors
            tail_entities, triplets = query(current_node.entity_id, source)

            # Filter valid triplets using rule-based filtering
            valid_triplets = []
            for triplet in triplets:
                try:
                    # Skip blacklisted predicates
                    if triplet[1].lower() in PREDICATE_BLACKLIST:
                        continue
                        
                    # Convert triplet to string format for filtering
                    triplet_str = f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]'
                    if filter_triple(triplet_str):  # Apply rule-based filtering
                        valid_triplets.append(triplet)
                except Exception as e:
                    print(f"Error filtering triplet: {e}")
                    continue

            # If a predicate occurs more than once, remove *all* triplets with that predicate
            predicate_counts = Counter(t[1] for t in valid_triplets)
            unique_predicate_triplets = [
                t for t in valid_triplets
                if predicate_counts[t[1]] == 1
            ]

            # Randomly select a subset of valid triplets
            valid_triplets = unique_predicate_triplets
            random.shuffle(valid_triplets)
            selected_triplets = valid_triplets[:num_neighbors_per_hop]

            # Process selected triplets and build the graph
            for triplet in selected_triplets:
                s, p, o = triplet
                # Match the object name with its entity ID
                matched = next(((e_id, e_name) for e_id, e_name in tail_entities if e_name == o), None)
                if not matched:
                    continue  # Skip if no match is found

                e_id, e_name = matched
                # Create or retrieve the target node
                if e_id in all_nodes:
                    new_node = all_nodes[e_id]
                else:
                    new_node = GraphNode(e_id, e_name)
                    all_nodes[e_id] = new_node
                    # Only add to next_nodes for expansion if:
                    # 1. We are not at the last hop, and
                    # 2. The predicate is not "instance of"
                    if current_hop < hops - 1 and p.lower() != "instance of":
                        next_nodes.append(new_node)

                # Add an edge from the current node to the new node
                current_node.add_edge(new_node, p)

        # Update the list of nodes to expand for the next hop
        nodes_to_expand = next_nodes
    
    return root

def extract_triplets_from_graph(root):
    """
    Extract all triples from the graph structure using BFS traversal.
    
    Args:
        root (GraphNode): The root node of the graph.
    
    Returns:
        list: A list of triples in the format [(subject, predicate, object)].
    """
    triplets = []  # List to store extracted triples
    visited = set()  # Set to track visited nodes
    stack = [root]  # Stack for BFS traversal
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        
        # Add all outgoing edges as triples
        for target, edge_label in node.outgoing_edges:
            triplets.append((node.name, edge_label, target.name))
            if target not in visited:
                stack.append(target)
    
    return triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="", help="entity id")
    parser.add_argument("--hops", type=int, default=1, help="number of hops")
    parser.add_argument("--ratio", type=float, default=1.0, help="ratio of neighbors to explore in the next hop (applies only to neighbors beyond the first hop; default: 1.0)")
    parser.add_argument("--save", action="store_true", help="enable GUI to save selected triples by clicking")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"], help="data source (wikidata or yago)")
    parser.add_argument("--multiple_samples", action="store_true", help="generate multiple samples")
    parser.add_argument("--num_samples", type=int, default=10, help="number of samples to generate (required if --multiple_samples is set)")
    parser.add_argument("--max_hops", type=int, default=3, help="maximum number of hops for each sample (required if --multiple_samples is set)")
    parser.add_argument("--parallel", action="store_true", help="enable parallel extraction")
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads for parallel extraction (required if --parallel is set)")
    parser.add_argument("--controlled_extraction", action="store_true", help="Enable controlled extraction mode, which selects and validates a specified number of neighbors per hop. Requires --num_neighbors_per_hop to be set.")
    parser.add_argument("--num_neighbors_per_hop", type=int, default=3, help="Number of neighbors to retrieve per hop in controlled mode")
    parser.add_argument("--type_qid", type=str, default=None, help="QID for entity type restriction (e.g., Q5 for humans)")
    parser.add_argument("--resume_generation", action="store_true",
                        help="Resume generation mode: extract only new samples, excluding entities that already have generated output files.")

    args = parser.parse_args()

    if args.multiple_samples:
        if args.num_samples <= 0:
            print("Number of samples must be greater than 0. Exiting.")
            sys.exit()
        if args.max_hops <= 0:
            print("Maximum number of hops must be greater than 0. Exiting.")
            sys.exit()
        if args.parallel and args.num_threads <= 0:
            print("Number of threads must be greater than 0. Exiting.")
            sys.exit()
        if args.controlled_extraction and args.num_neighbors_per_hop <= 0:
            print("Number of neighbors per hop must be greater than 0. Exiting.")
            sys.exit()

        # Process multiple samples
        process_multiple_samples(
            args.num_samples, args.max_hops, args.source, args.ratio, 
            args.parallel, args.num_threads, args.controlled_extraction, 
            args.num_neighbors_per_hop, args.type_qid, args.resume_generation
        )
    else:
        if not args.entity:
            # If no entity is provided, show the entity selection GUI based on the source
            if args.source == "wikidata":
                selected_entity = select_entity_from_wikidata(args.type_qid)
            elif args.source == "yago":
                selected_entity = select_entity_from_yago()
            else:
                print("Invalid source. Exiting.")
                sys.exit()

            if not selected_entity:
                print("No entity selected. Exiting.")
                sys.exit()
            args.entity = selected_entity[0].split('/')[-1]  # Extract entity ID from URI

        tail_entities, triplets = get_n_hop_neighbors(args.entity, args.hops, args.source, args.ratio)
        triplets = list(triplets)  # Convert to list for consistent ordering

        print('\nTail Entity IDs: ', [e[0] for e in tail_entities])
        print('\nTail Entity Names: ', [e[1] for e in tail_entities])

        print("\nTriplets:")
        for triplet in triplets:
            print(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]')

        if args.save:
            if not triplets:
                print("\nNo triples available to save.")
                sys.exit()
            create_gui(triplets)
