"""Neo4j Knowledge Graph Store for Dinesh's Career Profile (GraphRAG)."""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logger = logging.getLogger(__name__)

class ProfileGraphStore:
    """Manages connection, schema creation, ingestion, and querying in Neo4j Graph Database."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.driver = None
        self.is_connected = False
        self._init_driver()

    def _init_driver(self):
        """Initializes Neo4j Python driver."""
        if not self.password:
            logger.info("Neo4j password not configured; running in graph-offline / fallback mode.")
            return

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self.is_connected = True
            logger.info(f"Connected successfully to Neo4j Graph Database at {self.uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j ({e}). GraphRAG queries will fall back to local knowledge base.")
            self.driver = None
            self.is_connected = False

    def close(self):
        """Closes Neo4j connection pool."""
        if self.driver:
            self.driver.close()
            self.is_connected = False

    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a Cypher query against the Neo4j database."""
        if not self.is_connected or not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []

    def populate_knowledge_graph(self):
        """Populates Dinesh's career entities and relationships into Neo4j."""
        if not self.is_connected:
            logger.info("Neo4j not connected; skipping remote graph population.")
            return False

        cypher_statements = [
            # 1. Person Node (Dinesh)
            """
            MERGE (p:Person {name: 'Veerapareddy Dinesh'})
            ON CREATE SET 
                p.email = 'dineshveerapareddy@gmail.com',
                p.phone = '+91-8374330906',
                p.title = 'AI Engineer | GenAI Full Stack | Forward Deployed Engineer',
                p.location = 'Bengaluru, Karnataka, India',
                p.notice_period = '15 days',
                p.available_to_join = '15 days',
                p.portfolio = 'https://www.veerapareddydinesh.cyou/',
                p.linkedin = 'https://www.linkedin.com/in/dinesh-veerapareddy/',
                p.github = 'https://github.com/Dinesh-78',
                p.total_experience = '2.5+ years production (~3 years)'
            """,

            # 2. Company Nodes
            """
            MERGE (c1:Company {name: 'Carelon Global Solutions India'})
            ON CREATE SET c1.location = 'Bengaluru, Karnataka', c1.domain = 'Enterprise Healthcare'

            MERGE (c2:Company {name: 'Prajwal Fintech Private Limited'})
            ON CREATE SET c2.location = 'Hyderabad', c2.domain = 'Financial Technology'

            MERGE (c3:Company {name: 'MOVE37 Productions'})
            ON CREATE SET c3.location = 'Remote', c3.domain = 'AI Media Productions'
            """,

            # 3. Employment Relationships
            """
            MATCH (p:Person {name: 'Veerapareddy Dinesh'})
            MATCH (c1:Company {name: 'Carelon Global Solutions India'})
            MATCH (c2:Company {name: 'Prajwal Fintech Private Limited'})
            MATCH (c3:Company {name: 'MOVE37 Productions'})

            MERGE (p)-[r1:CURRENTLY_WORKS_AT {
                role: 'Associate Software Engineer I',
                period: 'Feb 2026 – Present',
                responsibilities: 'Own Benefits and Insurance module end-to-end, cut API response times by 20%'
            }]->(c1)

            MERGE (p)-[r2:PREVIOUSLY_WORKED_AT {
                role: 'Software Engineer',
                period: 'Mar 2025 – Feb 2026',
                responsibilities: 'Led team on MERN stack, Kafka async processing, improved performance by 40%'
            }]->(c2)

            MERGE (p)-[r3:PREVIOUSLY_WORKED_AT {
                role: 'Full Stack Developer',
                period: 'May 2024 – Mar 2025',
                responsibilities: 'Built AI audio-transcription feature with Next.js and TypeScript'
            }]->(c3)
            """,

            # 4. Education Node & Relationship
            """
            MATCH (p:Person {name: 'Veerapareddy Dinesh'})
            MERGE (u:Institution {name: 'KL University'})
            ON CREATE SET u.degree = 'B.Tech in Computer Science and Engineering', u.duration = '2020 – 2024', u.cgpa = 8.9
            MERGE (p)-[:GRADUATED_FROM {cgpa: 8.9, graduation_year: 2024}]->(u)
            """,

            # 5. Certifications
            """
            MATCH (p:Person {name: 'Veerapareddy Dinesh'})
            MERGE (cert1:Certification {name: 'AWS Certified Cloud Practitioner'})
            MERGE (cert2:Certification {name: 'Wipro Java Full Stack Certification'})
            MERGE (p)-[:EARNED]->(cert1)
            MERGE (p)-[:EARNED]->(cert2)
            """,

            # 6. Skill Nodes & Relationships
            """
            MATCH (p:Person {name: 'Veerapareddy Dinesh'})
            
            // Backend & Distributed Systems
            FOREACH (s IN ['Java', 'Spring Boot', 'Microservices', 'Node.js', 'Apache Kafka', 'Redis', 'RESTful APIs', 'MySQL'] |
                MERGE (sk:Skill {name: s, category: 'Backend & Systems'})
                MERGE (p)-[:HAS_SKILL]->(sk)
            )

            // AI & GenAI Skills
            FOREACH (s IN ['RAG', 'Vector Databases', 'Prompt Engineering', 'AI Agents', 'Tool Calling', 'Speech-to-Text', 'FastAPI', 'Python'] |
                MERGE (sk:Skill {name: s, category: 'Generative AI'})
                MERGE (p)-[:HAS_SKILL]->(sk)
            )

            // Frontend & Cloud
            FOREACH (s IN ['Angular', 'Next.js', 'React.js', 'TypeScript', 'Tailwind CSS', 'AWS', 'GCP', 'Docker', 'GitHub Actions'] |
                MERGE (sk:Skill {name: s, category: 'Frontend & DevOps'})
                MERGE (p)-[:HAS_SKILL]->(sk)
            )
            """,

            # 7. Map Skills directly to Companies
            """
            MATCH (c1:Company {name: 'Carelon Global Solutions India'})
            MATCH (s1:Skill) WHERE s1.name IN ['Java', 'Spring Boot', 'Angular', 'AWS', 'Microservices', 'MySQL']
            MERGE (c1)-[:UTILIZES_TECH]->(s1)

            MATCH (c2:Company {name: 'Prajwal Fintech Private Limited'})
            MATCH (s2:Skill) WHERE s2.name IN ['Apache Kafka', 'Redis', 'Node.js', 'Docker', 'GCP', 'GitHub Actions']
            MERGE (c2)-[:UTILIZES_TECH]->(s2)

            MATCH (c3:Company {name: 'MOVE37 Productions'})
            MATCH (s3:Skill) WHERE s3.name IN ['Speech-to-Text', 'Next.js', 'TypeScript', 'Tailwind CSS']
            MERGE (c3)-[:UTILIZES_TECH]->(s3)
            """
        ]

        logger.info("Ingesting Dinesh's career graph into Neo4j...")
        for query in cypher_statements:
            self.execute_cypher(query)

        logger.info("Neo4j Knowledge Graph successfully populated!")
        return True

    def query_graph_context(self, question: str) -> str:
        """Retrieves structured graph facts relevant to the recruiter's inquiry."""
        if not self.is_connected:
            return self._fallback_graph_context()

        # Query company, roles, and connected skills
        cypher = """
        MATCH (p:Person)-[r]->(c:Company)
        OPTIONAL MATCH (c)-[:UTILIZES_TECH]->(s:Skill)
        RETURN 
            p.name AS person,
            p.title AS title,
            p.location AS location,
            p.notice_period AS notice_period,
            type(r) AS relationship,
            r.role AS role,
            r.period AS period,
            c.name AS company,
            collect(s.name) AS company_technologies
        """
        records = self.execute_cypher(cypher)
        if not records:
            return self._fallback_graph_context()

        lines = ["=== NEO4J GRAPH FACTS ==="]
        for rec in records:
            lines.append(
                f"- {rec.get('person')} | Role: {rec.get('role')} at {rec.get('company')} ({rec.get('period')}). "
                f"Technologies: {', '.join(rec.get('company_technologies', []))}"
            )
        return "\n".join(lines)

    def _fallback_graph_context(self) -> str:
        """High-density relational facts when Neo4j remote server is not connected."""
        return """=== KNOWLEDGE GRAPH ENTITY RELATIONS ===
- (Veerapareddy Dinesh)-[:CURRENTLY_WORKS_AT {role: 'Associate Software Engineer I', period: 'Feb 2026 – Present'}]->(Carelon Global Solutions India [Bengaluru])
  Technologies Used: Java, Spring Boot, Angular, Microservices, AWS, MySQL.
  Key Impact: Owns Benefits & Insurance module; reduced API latency by 20%.

- (Veerapareddy Dinesh)-[:PREVIOUSLY_WORKED_AT {role: 'Software Engineer', period: 'Mar 2025 – Feb 2026'}]->(Prajwal Fintech Private Limited [Hyderabad])
  Technologies Used: Apache Kafka, Redis Caching, Node.js, Docker, GCP, GitHub Actions.
  Key Impact: Improved API performance by 40%; led development team.

- (Veerapareddy Dinesh)-[:PREVIOUSLY_WORKED_AT {role: 'Full Stack Developer', period: 'May 2024 – Mar 2025'}]->(MOVE37 Productions [Remote])
  Technologies Used: Speech-to-Text / Audio Transcription, Next.js, TypeScript, Tailwind CSS.

- (Veerapareddy Dinesh)-[:GRADUATED_FROM {degree: 'B.Tech in Computer Science and Engineering', year: 2024, cgpa: 8.9}]->(KL University)

- (Veerapareddy Dinesh)-[:EARNED]->(AWS Certified Cloud Practitioner)
- (Veerapareddy Dinesh)-[:EARNED]->(Wipro Java Full Stack Certification)

- Core Availability: Notice Period = 15 days, Location = Bengaluru (Open to Remote / Relocation). Salary & compensation details are confidential and discussed directly during interview discussions.
"""

_graph_instance = None

def get_graph_store() -> ProfileGraphStore:
    """Singleton getter for Neo4j ProfileGraphStore."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ProfileGraphStore()
    return _graph_instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🕸️ Initializing Neo4j ProfileGraphStore...")
    graph = get_graph_store()
    print("Neo4j Connected:", graph.is_connected)
    if graph.is_connected:
        graph.populate_knowledge_graph()
    print("\nGraph Context:\n", graph.query_graph_context("Tell me about Dinesh's companies"))
