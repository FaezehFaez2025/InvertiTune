import sys
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON

# Endpoint URL
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

def get_results(endpoint_url, query):
    """Execute SPARQL query and return results."""
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return {}

def get_entity_name(entity_id):
    """Get the human-readable name for a Wikidata entity ID."""
    sparql_query_src_entity_label = """
    SELECT ?srcLabel
    WHERE 
    {
      BIND(wd:%s AS ?src) .
      ?src rdfs:label ?srcLabel .
      FILTER(LANG(?srcLabel) = "en")
    }""" % entity_id

    try:
        r = get_results(WIKIDATA_ENDPOINT, sparql_query_src_entity_label)['results']['bindings']
        if len(r) == 0:
            return None
        src_entity_name = r[0]['srcLabel']['value']
        return src_entity_name
    except Exception as e:
        print(f"Error getting entity name: {e}")
        return None

def query_entity_triples(entity_id):
    """Query all triples where the Wikidata entity is the subject."""
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

    try:
        results = get_results(WIKIDATA_ENDPOINT, sparql_query_relations)
        src_entity_name = get_entity_name(entity_id)

        if src_entity_name is None:
            return []

        triplets = [(src_entity_name, result['realrelationLabel']['value'], result['xLabel']['value']) 
                   for result in results["results"]["bindings"]]

        return triplets
    except Exception as e:
        print(f"Error querying relations: {e}")
        return []

def print_entity_triples(entity_id):
    """
    Print all triples where the given Wikidata entity ID is the subject (starting entity).
    
    Args:
        entity_id (str): The Wikidata entity ID to query (e.g., "Q6581097")
    """
    print(f"\n{'='*80}")
    print(f"WIKIDATA ENTITY TRIPLES EXTRACTION")
    print(f"{'='*80}")
    print(f"Entity ID: {entity_id}")
    
    # Get entity name
    entity_name = get_entity_name(entity_id)
    if not entity_name:
        print(f"❌ Error: Could not find entity name for ID '{entity_id}' in Wikidata")
        return
    
    print(f"Entity Name: {entity_name}")
    print(f"{'='*80}")
    
    # Query all relations for this entity
    try:
        triplets = query_entity_triples(entity_id)
        
        if not triplets:
            print(f"No triples found for entity '{entity_name}' ({entity_id})")
            return
        
        print(f"\nFound {len(triplets)} total triples:")
        print(f"{'='*80}")
        
        # Print all triples with numbering
        for i, triplet in enumerate(triplets, 1):
            subject, predicate, obj = triplet
            print(f"{i:4d}. [{subject}, {predicate}, {obj}]")
        
        print(f"\n{'='*80}")
        print(f"Total triples displayed: {len(triplets)}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Error querying entity: {e}")

def main():
    """Main function to handle command line arguments and execute queries."""
    parser = argparse.ArgumentParser(description="Retrieves all relationships where the specified entity is the subject.")
    parser.add_argument("entity", type=str, help="Wikidata entity ID to query (e.g., Q6581097)")
    
    args = parser.parse_args()
    
    # Execute query
    print_entity_triples(args.entity)

if __name__ == "__main__":
    main()
