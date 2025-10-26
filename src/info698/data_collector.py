#!/usr/bin/env python3

import requests
import logging
import simplejson as json
import time
from typing import Dict, List, Any, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAlexAPI:
    def __init__(self, query, max_retries: int = 3, delay: float = 1.0):
        self.base_url = "https://api.openalex.org/works"
        self.headers = {
            'Accept': 'application/json',
        }
        self.query = query
        self.max_retries = max_retries
        self.delay = delay
        self.query_alex_repsone = None
        self.cites = None
        self.citation_url = None
        self.request_count = 0
        self.start_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a request with retry logic and rate limiting."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                if self.request_count > 0:
                    time.sleep(self.delay)
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Failed after {self.max_retries} attempts: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.delay * (2 ** attempt))
        
        raise Exception(f"All {self.max_retries} attempts failed")

    def get_openalex_id(self, page=1, per_page=25):
        """
        Fetch OpenAlex IDs based on a query with enhanced parameters.
        """
        params = {
            'search': self.query,
            'page': page,
            'per-page': per_page,
            'sort': 'cited_by_count:desc'  # Sort by citation count
        }
        
        logger.info(f"Searching for: {self.query}")
        response = self._make_request(self.base_url, params)
        data = response.json()
        
        if not data.get('results'):
            logger.warning(f"No results found for query: {self.query}")
            self.query_alex_repsone = None
            return None
        
        # Get the most cited result
        self.query_alex_repsone = data['results'][0]
        logger.info(f"Found paper: {self.query_alex_repsone.get('title', 'Unknown')}")
        return self.query_alex_repsone
    
    def get_citation_url(self):
        """
        Get the citation URL for a given OpenAlex ID.
        """
        if self.query_alex_repsone:
            self.citation_url = self.query_alex_repsone.get('cited_by_api_url', None)

        else:
            logger.warning(f"No OpenAlex ID found for query: {self.query}")
            self.citation_url = None
            raise Exception(f"No OpenAlex ID found for query: {self.query}")

    def query_citation_url(self, max_citations: int = 100):
        """
        Fetch citation data from OpenAlex using a citation URL with pagination.
        """
        if not self.citation_url:
            logger.error("No citation URL available")
            return []
        
        all_citations = []
        page = 1
        per_page = 25
        
        logger.info(f"Fetching citations from: {self.citation_url}")
        
        while len(all_citations) < max_citations:
            params = {
                'page': page,
                'per-page': per_page,
                'sort': 'cited_by_count:desc'
            }
            
            try:
                response = self._make_request(self.citation_url, params)
                data = response.json()
                
                if not data.get('results'):
                    logger.info(f"No more citations found at page {page}")
                    break
                
                citations = data.get('results', [])
                all_citations.extend(citations)
                
                logger.info(f"Fetched {len(citations)} citations from page {page} (total: {len(all_citations)})")
                
                # Check if we've reached the end
                if len(citations) < per_page:
                    break
                    
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching citations from page {page}: {e}")
                break
        
        self.cites = all_citations[:max_citations]
        logger.info(f"Total citations collected: {len(self.cites)}")
        return self.cites

    def get_citations(self, max_citations: int = 100, include_abstracts: bool = False):
        """
        Fetch citations, related works and references with enhanced data collection.
        """
        try:
            # Get the main paper
            self.get_openalex_id()
            if not self.query_alex_repsone:
                logger.error("Could not find the main paper")
                return {}
            
            # Get citation URL
            self.get_citation_url()
            if not self.citation_url:
                logger.error("Could not get citation URL")
                return {}
            
            # Fetch citations with pagination
            self.query_citation_url(max_citations)
            
            if not self.cites:
                logger.warning("No citations found")
                return {self.query_alex_repsone.get('id', "root"): {}}
            
            # Process citations with enhanced data (robust to malformed items)
            citations = {}
            for cite in self.cites:
                if not isinstance(cite, dict):
                    logger.warning("Skipping non-dict citation entry")
                    continue
                try:
                    related = cite.get('related_works')
                    if not isinstance(related, list):
                        related = []
                    refs = cite.get('referenced_works')
                    if not isinstance(refs, list):
                        refs = []
                    authorships = cite.get('authorships')
                    if not isinstance(authorships, list):
                        authorships = []

                    citation_data = {
                        'title': cite.get('title', 'Unknown Title'),
                        'openalex_id': cite.get('id', None),
                        'cited_by_count': cite.get('cited_by_count', 0),
                        'publication_year': cite.get('publication_year', None),
                        'related_works': related[:10],  # Increased from 5
                        'references': refs[:10],        # Increased from 5
                        'authors': [
                            (
                                # Prefer nested author.display_name (OpenAlex schema)
                                (author.get('author') or {}).get('display_name')
                                or author.get('raw_author_name')
                                or ( (author.get('institutions') or [{}])[0].get('display_name') if isinstance(author.get('institutions'), list) and author.get('institutions') else None )
                                or ''
                            ) if isinstance(author, dict) else ''
                            for author in authorships
                        ],
                        'venue': (cite.get('primary_location') or {}).get('source', {}).get('display_name', ''),
                        'doi': cite.get('doi'),
                        'concepts': [
                            (c.get('display_name', '') if isinstance(c, dict) else '')
                            for c in (cite.get('concepts') or [])
                        ],
                        'type': cite.get('type', 'journal-article'),
                        'language': cite.get('language', 'en'),
                        'is_oa': (cite.get('open_access') or {}).get('is_oa', False),
                        'oa_url': (cite.get('open_access') or {}).get('oa_url', None)
                    }

                    # Include abstract if requested
                    if include_abstracts:
                        inv = cite.get('abstract_inverted_index') or {}
                        if isinstance(inv, dict) and inv:
                            citation_data['abstract'] = self._reconstruct_abstract(inv)

                    key = cite.get('id') or f"unknown_{len(citations)}"
                    citations[key] = citation_data
                except Exception as item_err:
                    logger.warning(f"Skipping malformed citation entry due to error: {item_err}")
                    continue
            
            # Add metadata about the collection
            collection_metadata = {
                'query': self.query,
                'total_citations': len(citations),
                'collection_time': time.time() - self.start_time,
                'requests_made': self.request_count,
                'main_paper': {
                    'title': self.query_alex_repsone.get('title', ''),
                    'id': self.query_alex_repsone.get('id', ''),
                    'cited_by_count': self.query_alex_repsone.get('cited_by_count', 0),
                    'publication_year': self.query_alex_repsone.get('publication_year', None)
                }
            }
            
            result = {
                self.query_alex_repsone.get('id', "root"): citations,
                '_metadata': collection_metadata
            }
            
            logger.info(f"Successfully collected {len(citations)} citations in {collection_metadata['collection_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in get_citations: {e}")
            return {}
    
    def _reconstruct_abstract(self, abstract_inverted_index: Dict) -> str:
        """Reconstruct abstract from inverted index format."""
        if not abstract_inverted_index:
            return ""
        
        # Flatten the inverted index
        words = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        
        # Sort by position and join
        words.sort(key=lambda x: x[0])
        return " ".join([word for _, word in words])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the data collection."""
        return {
            'requests_made': self.request_count,
            'collection_time': time.time() - self.start_time,
            'avg_request_time': (time.time() - self.start_time) / max(self.request_count, 1),
            'query': self.query,
            'main_paper_found': self.query_alex_repsone is not None,
            'citations_found': len(self.cites) if self.cites else 0
        }


if __name__ == "__main__":
    # Enhanced data collection with multiple queries
    queries = [
        "Attention is all you need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT: Generative Pre-trained Transformer"
    ]
    
    all_citations = {}
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Collecting data for: {query}")
        print(f"{'='*60}")
        
        try:
            # Initialize API with enhanced settings
            openalex_api = OpenAlexAPI(query, max_retries=3, delay=1.0)
            
            # Collect citations with enhanced data
            citations = openalex_api.get_citations(
                max_citations=50,  # Limit for demo
                include_abstracts=True  # Set to True if you want abstracts
            )
            
            if citations:
                all_citations.update(citations)
                
                # Show performance stats
                stats = openalex_api.get_performance_stats()
                print(f"📊 Performance Stats:")
                print(f"   Requests made: {stats['requests_made']}")
                print(f"   Collection time: {stats['collection_time']:.2f}s")
                print(f"   Citations found: {stats['citations_found']}")
                print(f"   Main paper found: {stats['main_paper_found']}")
                
                # Show metadata if available
                if '_metadata' in citations:
                    metadata = citations['_metadata']
                    print(f"📈 Collection Metadata:")
                    print(f"   Total citations: {metadata['total_citations']}")
                    print(f"   Collection time: {metadata['collection_time']:.2f}s")
                    print(f"   Main paper: {metadata['main_paper']['title']}")
            else:
                print(f"❌ No citations found for: {query}")
                
        except Exception as e:
            print(f"❌ Error collecting data for '{query}': {e}")
            continue
    
    # Save all collected data locally, not needing to call openalex again  - better for dev
    if all_citations:
        with open("./data/citations.json", "w") as _file:
            json.dump(all_citations, _file, indent=4)
        
        print(f"\n✅ Successfully saved {len(all_citations)} citation collections to citations.json")
        
        # Show summary
        total_citations = sum(len(data) for key, data in all_citations.items() if key != '_metadata')
        print(f"📊 Summary:")
        print(f"   Total papers collected: {total_citations}")
        print(f"   Root papers: {len([k for k in all_citations.keys() if k != '_metadata'])}")
    else:
        print("❌ No data collected")