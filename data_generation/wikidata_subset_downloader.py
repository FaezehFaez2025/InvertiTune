import os
import requests
from datetime import datetime
import argparse

SUBSET_REFERENCE = """
QID      Entity Type Description
Q5       Human (individual people)
Q12136   Disease (medical conditions and disorders)
Q16521   Taxon (biological taxa, including species and higher classifications)
Q7187    Gene (genetic elements)
Q11173   Chemical compound (chemical substances)
Q79007   Medical procedure (medical interventions and treatments)
Q4022    River (natural flowing watercourses)
Q8502    Mountain (natural landforms, peaks, and ranges)
Q515     City (urban settlements)
Q6256    Country (sovereign states)
Q11424   Film (motion pictures)
Q13442814 Scientific article (academic publications)
Q482994   Artist (creative professionals)
Q732577   Publication (published works)
Q3305213  Painting (visual artworks)
Q28389    Company (business organizations)
Q3918     University (higher education institutions)
Q1190554  Historical event (significant past occurrences)
Q7397     Software (computer programs)
Q386724   Website (online resources)
"""

def download_wikidata_subset(qid, limit):
    """Download RDF data for entities of a specific type from Wikidata"""
    today = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"wikidata_{qid}_{today}"
    os.makedirs(folder_name, exist_ok=True)

    sparql_query = f"""
    CONSTRUCT {{
      ?entity ?p ?o .
    }}
    WHERE {{
      ?entity wdt:P31/wdt:P279* wd:{qid} .
      ?entity ?p ?o .
    }}
    LIMIT {limit}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "Wikidata Downloader/1.0",
        "Accept": "application/rdf+xml"  # Changed to RDF/XML
    }

    try:
        print(f"Downloading {qid} subset from Wikidata (Limit: {limit})...")
        response = requests.get(
            url,
            headers=headers,
            params={"query": sparql_query, "format": "xml"},
            stream=True
        )
        response.raise_for_status()

        file_path = os.path.join(folder_name, f"{qid}_subset.rdf")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"\nSuccessfully downloaded to: {file_path}")
        print(f"File size: {os.path.getsize(file_path)//1024} KB")

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download specific Wikidata subsets",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-q", "--qid", type=str, default="Q5",
                      help=f"Wikidata QID for entity type\n{SUBSET_REFERENCE}")
    parser.add_argument("-l", "--limit", type=int, default=10000,
                      help="Maximum number of entities to download (default: 10000)")
    args = parser.parse_args()
    
    download_wikidata_subset(args.qid, args.limit)