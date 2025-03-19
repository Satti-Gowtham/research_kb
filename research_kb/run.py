import pytz
from typing import Dict, Any
from datetime import datetime
import json
import uuid
from naptha_sdk.schemas import KBRunInput
from naptha_sdk.storage.storage_client import StorageClient
from naptha_sdk.storage.schemas import (
    CreateStorageRequest, 
    ReadStorageRequest, 
    DeleteStorageRequest,
    ListStorageRequest
)
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from naptha_sdk.utils import get_logger
from dotenv import load_dotenv

from research_kb.schemas import InputSchema

load_dotenv()

logger = get_logger(__name__)

class ResearchKB:
    """Research Knowledge Base for storing research findings."""
    
    def __init__(self, deployment: Dict[str, Any]):
        self.storage_provider = StorageClient(deployment.node)
        
        # Set up storage options
        storage_config = deployment.config.storage_config
        self.storage_type = storage_config.storage_type
        self.table_name = storage_config.path
        self.research_schema = storage_config.storage_schema
        
    async def initialize(self, *args, **kwargs) -> Dict[str, Any]:
        """Initialize the research knowledge base"""
        try:
            # Check if table exists, create if not
            if not await self.table_exists(self.table_name):
                logger.info(f"Creating table: {self.table_name}")
                # Create research table
                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.table_name,
                    data={"schema": self.research_schema}
                )
                await self.storage_provider.execute(create_request)
                
            return {"status": "success", "message": "Research knowledge base initialized"}
        except Exception as e:
            logger.error(f"Error initializing research knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            list_request = ListStorageRequest(
                storage_type=self.storage_type,
                path=table_name
            )
            await self.storage_provider.execute(list_request)
            return True
        except Exception:
            return False

    async def ingest_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Store research findings in the knowledge base """
        try:
            await self.initialize()
            
            knowledge_data = {
                "run_id": input_data.get("run_id"),
                "topic": input_data.get("topic"),
                "content": input_data.get("content"),
                "agent_id": input_data.get("agent_id"),
                "round": input_data.get("round"),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }

            create_request = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data={"data": knowledge_data}
            )

            result = await self.storage_provider.execute(create_request)
            
            if not result.data:
                return {"status": "error", "message": "Failed to store research finding"}
                
            return {
                "status": "success",
                "id": knowledge_data["run_id"]
            }
            
        except Exception as e:
            logger.error(f"Error storing research finding: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_findings(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Retrieve research findings based on filters """
        try:
            await self.initialize()
            
            read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"conditions": [input_data]}
            )
            
            results = await self.storage_provider.execute(read_request)
            
            return {
                "status": "success",
                "data": results.data
            }
            
        except Exception as e:
            logger.error(f"Error retrieving research findings: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_by_id(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Retrieve a specific research finding by ID """
        try:
            read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"conditions": [{"run_id": input_data["run_id"]}]}
            )
            result = await self.storage_provider.execute(read_request)
            return {"status": "success", "data": result.data}
        except Exception as e:
            logger.error(f"Error retrieving research finding by ID: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def clear(self, *args, **kwargs) -> Dict[str, Any]:
        """ Clear all data from the knowledge base """
        try:
            delete_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={}
            )
            await self.storage_provider.execute(delete_request)
            
            return {"status": "success", "message": "Knowledge base cleared"}
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """ Run the Research Knowledge Base deployment """
    try:
        module_run = KBRunInput(**module_run)
        module_run.inputs = InputSchema(**module_run.inputs)
        research_kb = ResearchKB(module_run.deployment)

        method = getattr(research_kb, module_run.inputs.func_name, None)
        if not method:
            raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

        result = await method(module_run.inputs.func_input_data)
        return {"status": "success", "results": [json.dumps(result)]}
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

    async def main():
        """Example of how to use the Research KB module"""
        if not os.getenv("NODE_URL"):
            print("Please set NODE_URL environment variable to run this script directly")
            return

        try:
            # Initialize Naptha client
            naptha = Naptha()
            
            # Set up deployment
            deployment = await setup_module_deployment(
                "kb",
                "research_kb/configs/deployment.json",
                node_url=os.getenv("NODE_URL")
            )

            # Create test data
            test_data = {
                "run_id": "test_run_001",
                "topic": "AI in Healthcare",
                "content": "AI is revolutionizing healthcare by improving diagnosis accuracy and treatment planning.",
                "agent_id": "agent_1",
                "round": 1
            }

            # Test 1: Initialize KB
            print("\n1. Testing KB initialization...")
            init_result = await run({
                "deployment": deployment,
                "consumer_id": naptha.user.id,
                "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
                "inputs": {
                    "func_name": "initialize",
                    "func_input_data": {}
                }
            })
            print(f"Initialization result: {json.dumps(init_result, indent=2)}")

            # Test 2: Store research finding
            print("\n2. Testing research finding storage...")
            store_result = await run({
                "deployment": deployment,
                "consumer_id": naptha.user.id,
                "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
                "inputs": {
                    "func_name": "ingest_knowledge",
                    "func_input_data": test_data
                }
            })
            print(f"Storage result: {json.dumps(store_result, indent=2)}")

            # Test 3: Retrieve finding by ID
            if store_result["status"] == "success":
                stored_id = json.loads(store_result["results"][0])["id"]
                print("\n3. Testing retrieval by ID...")
                retrieve_result = await run({
                    "deployment": deployment,
                    "consumer_id": naptha.user.id,
                    "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
                    "inputs": {
                        "func_name": "get_by_id",
                        "func_input_data": {
                            "run_id": stored_id
                        }
                    }
                })
            print(f"Retrieval result: {json.dumps(retrieve_result, indent=2)}")

            # Test 4: Get findings with filters
            print("\n4. Testing filtered retrieval...")
            filter_result = await run({
                "deployment": deployment,
                "consumer_id": naptha.user.id,
                "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
                "inputs": {
                    "func_name": "get_findings",
                    "func_input_data": {
                        "run_id": 'test_run_001',
                        "topic": 'AI in Healthcare'
                    }
                }
            })
            print(f"Filtered retrieval result: {json.dumps(filter_result, indent=2)}")

            # Test 5: Clear KB
            print("\n5. Testing KB clearing...")
            clear_result = await run({
                "deployment": deployment,
                "consumer_id": naptha.user.id,
                "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
                "inputs": {
                    "func_name": "clear",
                    "func_input_data": {}
                }
            })
            print(f"Clear result: {json.dumps(clear_result, indent=2)}")

            print("\nAll tests completed successfully!")

        except Exception as e:
            print(f"\nError during testing: {str(e)}")
            raise e

    asyncio.run(main())