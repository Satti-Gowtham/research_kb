from dotenv import load_dotenv
from research_kb.schemas import InputSchema
from typing import Dict, Any, List
from naptha_sdk.schemas import KBRunInput, KBDeployment
from naptha_sdk.storage.schemas import CreateStorageRequest, ReadStorageRequest, ListStorageRequest, DeleteStorageRequest, DatabaseReadOptions
from naptha_sdk.storage.storage_client import StorageClient
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from research_kb.utils.embeddings import OllamaEmbedder
from research_kb.utils.chunker import SemanticChunker
import json
from datetime import datetime
import pytz

load_dotenv()
logger = get_logger(__name__)

class ResearchKB:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_client = StorageClient(self.deployment.node)
        self.storage_type = self.config.storage_config.storage_type
        self.table_name = self.config.storage_config.path
        self.schema = self.config.storage_config.storage_schema
        self.chunks_table = f"{self.table_name}_chunks"
        self.chunks_schema = {
            "id": {"type": "INTEGER", "primary_key": True},
            "run_id": {"type": "TEXT"},
            "text": {"type": "TEXT"},
            "embedding": {"type": "VECTOR", "dimension": 768},
            "start": {"type": "INTEGER"},
            "ends_at": {"type": "INTEGER"},
            "content_type": {"type": "TEXT"},
        }
        self.embedder = OllamaEmbedder(
            model=self.config.llm_config.model,
            url=self.config.llm_config.api_base
        )
        self.chunker = SemanticChunker()

    async def init(self, *args, **kwargs):
        """Initialize the knowledge base tables."""
        try:
            # Create main table
            create_request = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"schema": self.schema}
            )
            await self.storage_client.execute(create_request)
            
            # Create chunks table
            chunks_request = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                options={"schema": self.chunks_schema}
            )
            await self.storage_client.execute(chunks_request)
            
            return {"status": "success", "message": f"Successfully initialized tables {self.table_name} and {self.chunks_table}"}
        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):
        """Add data to the knowledge base with embeddings."""
        await self.init()
        try:
            logger.info(f"Adding data to table {self.table_name}")
            # Prepare findings text
            findings_text = " ".join([
                f"{finding.get('section', '')}: {', '.join(finding.get('points', []))}"
                for finding in input_data.get('findings', [])
            ])

            # Add timestamp
            input_data['timestamp'] = datetime.now(pytz.UTC).isoformat()

            create_row_result = await self.storage_client.execute(CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data={"data": input_data}
            ))

            # Create chunks for findings
            if findings_text:
                findings_chunks = self.chunker.chunk(findings_text)
                for chunk in findings_chunks:
                    chunk_embedding = await self.embedder.embed_text(chunk["text"])
                    chunk_data = {
                        "run_id": input_data["run_id"],
                        "text": chunk["text"],
                        "embedding": chunk_embedding,
                        "start": chunk["start"],
                        "ends_at": chunk["end"],
                        "content_type": "findings",
                        "metadata": {
                            "round": input_data.get("metadata", {}).get("round", 0),
                            "section": "findings"
                        }
                    }
                    await self.storage_client.execute(CreateStorageRequest(
                        storage_type=self.storage_type,
                        path=self.chunks_table,
                        data={"data": chunk_data}
                    ))

            logger.info(f"Successfully added data to table {self.table_name}")
            return {"status": "success", "message": f"Successfully added data to table {self.table_name}"}
            
        except Exception as e:
            logger.error(f"Error adding data: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_relevant_context(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant context based on semantic similarity to the query."""
        await self.init()
        try:
            # Get query embedding
            query = input_data["query"]
            query_embedding = await self.embedder.embed_text(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []

            # Set up vector search options
            chunks_db_options = DatabaseReadOptions(
                query_vector=query_embedding,
                query_col="embedding",
                answer_col="text",
                top_k=input_data.get("limit", 3),
                include_similarity=True
            )

            # Search chunks
            chunks_read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                options=chunks_db_options.model_dump()
            )
            
            chunk_results = await self.storage_client.execute(chunks_read_request)
            
            if not chunk_results.data:
                return []

            # Calculate proper similarity scores and filter results
            filtered_chunks = []
            for chunk in chunk_results.data:
                if "embedding" in chunk:
                    emb = chunk["embedding"]
                    if isinstance(emb, str):
                        try:
                            if emb.startswith('[') and emb.endswith(']'):
                                emb = json.loads(emb)
                            else:
                                emb = [float(x) for x in emb.split(',')]
                        except Exception:
                            continue
                    elif isinstance(emb, (list, tuple)):
                        try:
                            emb = [float(x) for x in emb]
                        except Exception:
                            continue
                    elif hasattr(emb, 'tolist'):
                        try:
                            emb = emb.tolist()
                        except Exception:
                            continue
                            
                    if emb is not None:
                        similarity_score = self.embedder.calculate_similarity(
                            query_embedding, 
                            emb,
                            query=query,
                            text=chunk["text"]
                        )
                        
                        chunk["similarity_score"] = similarity_score
                        filtered_chunks.append(chunk)

            # Sort by similarity
            filtered_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            run_ids = list(set(chunk["run_id"] for chunk in filtered_chunks))
            main_entries = await self.storage_client.execute(ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"condition": {"run_id": {"$in": run_ids}}}
            ))

            # Combine results
            results = []
            for chunk in filtered_chunks:
                for entry in main_entries.data:
                    if chunk["run_id"] == entry["run_id"]:
                        results.append({
                            'similarity': chunk['similarity_score'],
                            'findings': entry.get('findings', []),
                            'metadata': entry.get('metadata', {}),
                            'chunk': chunk['text']
                        })
                        break

            return results[:input_data.get("limit", 3)]
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return []

    async def list_rows(self, input_data: Dict[str, Any], *args, **kwargs):
        """List rows from the knowledge base."""
        await self.init()
        try:
            list_storage_request = ListStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"limit": input_data.get('limit') if input_data and 'limit' in input_data else None}
            )
            list_storage_result = await self.storage_client.execute(list_storage_request)
            return {"status": "success", "data": list_storage_result.data["data"]}
        except Exception as e:
            logger.error(f"Error listing rows: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        """Delete the knowledge base table."""
        try:
            delete_table_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=input_data['table_name'],
            )
            await self.storage_client.execute(delete_table_request)

            delete_chunks_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=f"{input_data['table_name']}_chunks",
            )
            await self.storage_client.execute(delete_chunks_request)
            return {"status": "success", "message": "Table deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting table: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict):
    """Main entry point for the module."""
    try:
        module_run = KBRunInput(**module_run)
        module_run.inputs = InputSchema(**module_run.inputs)
        research_kb = ResearchKB(module_run.deployment)
        method = getattr(research_kb, module_run.inputs.func_name, None)
        return await method(module_run.inputs.func_input_data)
    except Exception as e:
        logger.error(f"Error in run: {str(e)}")
        return {
            "status": "error",
            "error": True,
            "error_message": f"Error in run: {str(e)}"
        }

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("kb", "./research_kb/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    inputs_dict = {
        "adding_data": {
            "func_name": "add_data",
            "func_input_data": {
                "run_id": "123",
                "topic": "What are the implications of synthetic life?",
                "findings": [
                    {
                        "section": "findings",
                        "points": ["Point 1", "Point 2", "Point 3"]
                    }
                ],
                "metadata": {
                    "round": 1,
                    "topic": "What are the implications of synthetic life?"
                }
            }
        },
        "listing_rows": {
            "func_name": "list_rows",
            "func_input_data": {
                "limit": 10
            }
        },
        "deleting_table": {
            "func_name": "delete_table",
            "func_input_data": {
                "table_name": "research_kb"
            }
        },
        "getting_relevant_context": {
            "func_name": "get_relevant_context",
            "func_input_data": {
                "query": "What are the implications of synthetic life?"
            }
        },
        "initializing_tables": {
            "func_name": "init",
            "func_input_data": {}
        }
    }
    module_run = {
        "inputs": inputs_dict["initializing_tables"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))

    print("Response: ", response)