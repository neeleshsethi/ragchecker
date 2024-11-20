import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import boto3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd

@dataclass
class KnowledgeBase:
    """Class to define knowledge base configurations"""
    name: str
    kb_id: str  # Bedrock Knowledge Base ID
    chunk_size: int
    chunk_overlap: int
    description: Optional[str] = None

class BedrockKBEvaluator:
    """Class to evaluate RAG performance across different Bedrock knowledge bases"""
    
    def __init__(
        self,
        extractor_model: str = "bedrock/meta.llama3-1-70b-instruct-v1:0",
        checker_model: str = "bedrock/meta.llama3-1-70b-instruct-v1:0",
        batch_size_extractor: int = 32,
        batch_size_checker: int = 32,
        region_name: str = "us-east-1"
    ):
        self.evaluator = RAGChecker(
            extractor_name=extractor_model,
            checker_name=checker_model,
            batch_size_extractor=batch_size_extractor,
            batch_size_checker=batch_size_checker
        )
        self.bedrock = boto3.client('bedrock', region_name=region_name)
        self.bedrock_kb = boto3.client('bedrock-knowledge-bases', region_name=region_name)
        
    def query_knowledge_base(
        self,
        kb_id: str,
        query: str,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Query a specific Bedrock knowledge base
        """
        try:
            response = self.bedrock_kb.retrieve_and_generate(
                knowledgeBaseId=kb_id,
                input={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': 4  # Match the example's context length
                    }
                },
                generationConfiguration={
                    'maximumLength': max_tokens,
                    'temperature': temperature
                }
            )
            
            # Extract retrieved passages and generated response
            retrieved_context = [
                {
                    "doc_id": str(idx),
                    "text": passage['content']
                }
                for idx, passage in enumerate(response['retrievalResults'])
            ]
            
            return {
                "response": response['generationResult']['text'],
                "retrieved_context": retrieved_context
            }
            
        except Exception as e:
            print(f"Error querying knowledge base {kb_id}: {str(e)}")
            return None

    def prepare_rag_results(
        self,
        query_data: Dict[str, Any],
        kb: KnowledgeBase
    ) -> RAGResults:
        """
        Prepare RAG results for a specific knowledge base
        """
        modified_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for result in query_data["results"]:
                future = executor.submit(
                    self.query_knowledge_base,
                    kb.kb_id,
                    result["query"]
                )
                futures.append((result, future))
            
            for result, future in futures:
                kb_response = future.result()
                
                if kb_response:
                    modified_result = {
                        "query_id": result["query_id"],
                        "query": result["query"],
                        "gt_answer": result["gt_answer"],
                        "response": kb_response["response"],
                        "retrieved_context": kb_response["retrieved_context"]
                    }
                    modified_results.append(modified_result)
        
        return RAGResults.from_dict({"results": modified_results})

    def evaluate_knowledge_base(
        self,
        query_data: Dict[str, Any],
        kb: KnowledgeBase
    ) -> Dict[str, Any]:
        """
        Evaluate a single knowledge base
        """
        print(f"\nEvaluating knowledge base: {kb.name}")
        
        # Prepare RAG results
        rag_results = self.prepare_rag_results(query_data, kb)
        
        # Evaluate using RAGChecker
        self.evaluator.evaluate(rag_results, all_metrics)
        
        return {
            "kb_name": kb.name,
            "kb_id": kb.kb_id,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
            "description": kb.description,
            "metrics": rag_results.metrics,
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_multiple_kbs(
        self,
        query_data: Dict[str, Any],
        knowledge_bases: List[KnowledgeBase]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple knowledge bases and compare results
        """
        results = []
        for kb in knowledge_bases:
            evaluation = self.evaluate_knowledge_base(query_data, kb)
            results.append(evaluation)
        return results

    def generate_comparison_report(
        self,
        results: List[Dict[str, Any]],
        output_path: str = "kb_comparison_report.html"
    ) -> None:
        """
        Generate a detailed HTML report comparing knowledge base performances
        """
        # Convert metrics to a structured format for analysis
        metrics_data = []
        for result in results:
            kb_metrics = {
                "Knowledge Base": result["kb_name"],
                "Chunk Size": result["chunk_size"],
                "Chunk Overlap": result["chunk_overlap"]
            }
            
            # Flatten metrics structure
            for metric_group, metrics in result["metrics"].items():
                for metric_name, value in metrics.items():
                    kb_metrics[f"{metric_group}_{metric_name}"] = value
                    
            metrics_data.append(kb_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Knowledge Base Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-group {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Knowledge Base Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric-group">
                <h2>Configuration Comparison</h2>
                {df[["Knowledge Base", "Chunk Size", "Chunk Overlap"]].to_html(index=False)}
            </div>
            
            <div class="metric-group">
                <h2>Performance Metrics</h2>
                {df.drop(["Chunk Size", "Chunk Overlap"], axis=1).to_html(index=False)}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"\nComparison report generated: {output_path}")

def main():
    # Define your existing knowledge bases
    knowledge_bases = [
        KnowledgeBase(
            name="KB_Small_Chunks",
            kb_id="kb-1234567890",  # Replace with actual KB ID
            chunk_size=256,
            chunk_overlap=64,
            description="Knowledge base with small chunks"
        ),
        KnowledgeBase(
            name="KB_Medium_Chunks",
            kb_id="kb-2345678901",  # Replace with actual KB ID
            chunk_size=512,
            chunk_overlap=128,
            description="Knowledge base with medium chunks"
        ),
        KnowledgeBase(
            name="KB_Large_Chunks",
            kb_id="kb-3456789012",  # Replace with actual KB ID
            chunk_size=1024,
            chunk_overlap=256,
            description="Knowledge base with large chunks"
        ),
        KnowledgeBase(
            name="KB_Extra_Large_Chunks",
            kb_id="kb-4567890123",  # Replace with actual KB ID
            chunk_size=2048,
            chunk_overlap=512,
            description="Knowledge base with extra large chunks"
        )
    ]
    
    # Load your query data
    with open("examples/checking_inputs.json") as fp:
        query_data = json.load(fp)
    
    # Initialize evaluator
    evaluator = BedrockKBEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_multiple_kbs(query_data, knowledge_bases)
    
    # Generate comparison report
    evaluator.generate_comparison_report(results)
    
    # Print summary results
    print("\nEvaluation Summary:")
    print("==================")
    for result in results:
        print(f"\nKnowledge Base: {result['kb_name']}")
        print(f"Chunk Size: {result['chunk_size']}")
        print(f"Chunk Overlap: {result['chunk_overlap']}")
        print("\nMetrics:")
        for metric_group, metrics in result['metrics'].items():
            print(f"\n{metric_group}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")

if __name__ == "__main__":
    main()
