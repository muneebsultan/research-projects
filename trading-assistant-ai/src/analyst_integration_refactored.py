import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AnalystTaskConverter:
    """Converts tasks between agent and analyst system formats."""

    ANALYST_KEYWORDS = [
        # Analysis operations
        "analyze", "analysis", "calculate", "computation", "compute",
        # Financial terms
        "financial", "ratio", "metrics", "performance", "returns",
        "revenue", "profit", "margin", "growth", "valuation",
        # Classification
        "classify", "classification", "categorize", "categorisation",
        # Data analysis
        "outlier", "correlation", "trend", "pattern", "statistics",
        "regression", "forecast", "prediction", "model",
        # DataFrame operations
        "dataframe", "data analysis", "quantitative", "pandas",
        # Financial metrics
        "ROIC", "ROE", "ROA", "EBITDA", "P/E", "debt", "equity",
        "cash flow", "balance sheet", "income statement",
        # Force keywords
        "collection:", "redis", "dividend", "yields"
    ]

    DATA_PHRASES = [
        "detect outliers", "calculate ratios", "find correlation",
        "analyze trends", "compare performance", "financial analysis",
        "data mining", "statistical analysis"
    ]

    ANALYSIS_TYPE_MAPPING = {
        "outlier": "outlier_detection",
        "correlation": "correlation_analysis",
        "trend": "trend_analysis",
        "ratio": "financial_ratio_analysis",
        "metric": "financial_ratio_analysis",
        "performance": "performance_analysis",
    }

    @staticmethod
    def is_analyst_task(task: Any) -> bool:
        """
        Determine if a task should be routed to analyst system.

        Args:
            task: Task object with clean_task and description attributes

        Returns:
            bool: True if task should use analyst system
        """
        try:
            task_text = f"{task.clean_task} {task.description}".lower()

            # Check keyword and phrase matches
            keyword_match = any(
                keyword.lower() in task_text
                for keyword in AnalystTaskConverter.ANALYST_KEYWORDS
            )
            phrase_match = any(
                phrase in task_text
                for phrase in AnalystTaskConverter.DATA_PHRASES
            )
            force_analyst = "collection:" in task_text

            result = keyword_match or phrase_match or force_analyst

            logger.debug(
                f"Task '{task.clean_task}' - Keyword: {keyword_match}, "
                f"Phrase: {phrase_match}, Force: {force_analyst} → {result}"
            )

            return result

        except AttributeError as e:
            logger.error(f"Invalid task object: {e}")
            return False

    @staticmethod
    def extract_data_location(task: Any) -> Optional[str]:
        """
        Extract Redis collection ID from task description.

        Args:
            task: Task object with description

        Returns:
            Optional[str]: Redis collection ID or None
        """
        patterns = [
            r'collection:([^\s\n,]+)',
            r'redis_id:\s*([^\s\n,]+)',
            r'data_location:\s*([^\s\n,]+)',
            r'Redis URL:\s*([^\s\n]+)'
        ]

        search_text = f"{task.clean_task} {task.description}"

        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                redis_id = match.group(1).strip()
                if redis_id.startswith('collection:'):
                    redis_id = redis_id[11:]
                logger.info(f"Extracted data location: {redis_id}")
                return redis_id

        logger.warning(f"No data location found in task: {task.clean_task}")
        return None

    @staticmethod
    def task_to_analyst_format(task: Any) -> str:
        """
        Convert agent task to analyst task metadata format.

        Args:
            task: Agent task to convert

        Returns:
            str: Formatted task metadata
        """
        data_location = AnalystTaskConverter.extract_data_location(task)
        data_location_line = (
            f"Data Redis URL to access dataframe: collection:{data_location}"
            if data_location
            else "Data Redis URL to access dataframe: None"
        )

        task_metadata = f"""Your task is: {task.clean_task}

Task description: {task.description}
Symbol: {task.symbol}
{data_location_line}
Data description: Financial analysis task from agent workflow system
Data columns description: Analysis task routed from multi-agent system
Additional context: This task was identified as requiring specialized financial data analysis capabilities.
"""

        logger.info(f"Converted task {task.task_id} to analyst format")
        return task_metadata

    @staticmethod
    def create_analyst_task(task: Any) -> Dict[str, Any]:
        """
        Create analyst task from agent task.

        Args:
            task: Agent task to convert

        Returns:
            Dict with analyst task configuration
        """
        task_metadata = AnalystTaskConverter.task_to_analyst_format(task)
        data_location = AnalystTaskConverter.extract_data_location(task)
        
        # Determine analysis type
        analysis_type = "general_financial_analysis"
        task_lower = task.clean_task.lower()

        for keyword, analysis_key in AnalystTaskConverter.ANALYSIS_TYPE_MAPPING.items():
            if keyword in task_lower:
                analysis_type = analysis_key
                break

        return {
            "task_id": task.task_id,
            "task_metadata": task_metadata,
            "data_location": data_location,
            "symbol": task.symbol,
            "analysis_requested": analysis_type
        }


class AnalystResultFormatter:
    """Formats analyst results for various output contexts."""

    RESULT_TEMPLATE = """📊 **{analysis_title}**

{final_output}

**🔧 Tools Used**: {tools_used}
**💾 Data Storage**: {storage_info}
**📈 Result Type**: {result_type}
"""

    ERROR_TEMPLATE = """### ❌ {analysis_title} (Failed)

**Error**: {error_message}
"""

    @staticmethod
    def format_for_agent(analyst_result: Dict[str, Any]) -> str:
        """
        Format analyst result as agent output.

        Args:
            analyst_result: Result from analyst system

        Returns:
            str: Formatted output
        """
        final_output = analyst_result.get("final_output", "")
        redis_id = analyst_result.get("redis_id")
        error_message = analyst_result.get("error_message")

        if error_message:
            return f"❌ **Analysis Error**: {error_message}\n\n{final_output}"

        if redis_id:
            return AnalystResultFormatter._format_with_storage(
                final_output, redis_id
            )
        else:
            return f"📊 **Financial Analysis Complete**\n\n{final_output}"

    @staticmethod
    def _format_with_storage(final_output: str, redis_id: str) -> str:
        """Format result with Redis storage information."""
        return f"""📊 **Financial Data Analysis Complete**

{final_output}

**📁 Detailed Results Stored**: `{redis_id}`
**🔗 Access Pattern**: 
```python
from app.analyst.utils.redis_utils import RedisManager
redis_manager = RedisManager()
collection_data, success = redis_manager.retrieve_collection('{redis_id}')
if success:
    df = collection_data['dataframe']
    metadata = collection_data['metadata']
    tools_used = collection_data['tools_used']
```

**💡 Analysis Summary**: The detailed analysis results including any DataFrames, 
statistical outputs, and tool usage are permanently stored and can be retrieved 
for further processing or visualization."""

    @staticmethod
    def format_for_aggregator(analyst_results: List[Dict[str, Any]]) -> str:
        """
        Format multiple analyst results for aggregator.

        Args:
            analyst_results: List of analyst result dictionaries

        Returns:
            str: Formatted results string
        """
        if not analyst_results:
            return ""

        sections = []

        for result in analyst_results:
            if result.get("success", False):
                section = AnalystResultFormatter._format_success_result(result)
            else:
                section = AnalystResultFormatter._format_failed_result(result)

            sections.append(section)

        return "\n\n".join(sections)

    @staticmethod
    def _format_success_result(result: Dict[str, Any]) -> str:
        """Format successful analysis result."""
        analysis_type = result.get("analysis_type", "analysis").replace('_', ' ').title()
        tools_used = ", ".join(result.get("tools_used", []))
        storage_info = result.get("redis_id", "In-memory only")
        result_type = result.get("result_type", "unknown")

        section = f"""### 📊 {analysis_type}

{result.get('final_output', '')}

**🔧 Tools Used**: {tools_used or 'None'}
**💾 Data Storage**: {storage_info}
**📈 Result Type**: {result_type}
"""

        if result.get("dataframe_info"):
            df_info = result["dataframe_info"]
            shape = df_info.get("shape", "Unknown")
            cols = len(df_info.get("columns", []))
            section += f"**📋 DataFrame Info**: Shape {shape}, Columns: {cols}\n"

        return section

    @staticmethod
    def _format_failed_result(result: Dict[str, Any]) -> str:
        """Format failed analysis result."""
        analysis_type = result.get("analysis_type", "analysis").replace('_', ' ').title()
        error_msg = result.get("error_message", "Unknown error")

        return f"""### ❌ {analysis_type} (Failed)

**Error**: {error_msg}
"""


def update_task_with_analyst_result(task: Any, analyst_result: Dict[str, Any]) -> None:
    """
    Update agent task with analyst system results.

    Args:
        task: Agent task to update
        analyst_result: Result from analyst system
    """
    try:
        if analyst_result.get("success", False):
            task.status = "completed"
            task.output = AnalystResultFormatter.format_for_agent(analyst_result)
        else:
            task.status = "failed"
            task.error_message = analyst_result.get("error_message", "Unknown error")
            task.retry_count = getattr(task, 'retry_count', 0) + 1

        logger.info(
            f"Updated task {task.task_id} with analyst result. "
            f"Status: {task.status}"
        )

    except AttributeError as e:
        logger.error(f"Error updating task {task.task_id}: {e}")
