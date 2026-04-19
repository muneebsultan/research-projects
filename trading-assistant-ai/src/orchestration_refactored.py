import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MessageExtractor:
    """Extracts and processes messages from conversations."""

    SOURCE_MAPPING = {
        "work_flows": "Work Flows",
        "web_researcher": "Web Researcher",
        "internal_search": "Internal Search"
    }

    @staticmethod
    def get_messages(conversation: Union[Dict[str, Any], Any]) -> Optional[List[Any]]:
        """
        Extract messages from conversation object.

        Args:
            conversation: Conversation object or dict

        Returns:
            Optional[List]: Messages list or None
        """
        try:
            if hasattr(conversation, "messages"):
                return conversation.messages
            elif isinstance(conversation, dict) and "messages" in conversation:
                return conversation["messages"]
            else:
                logger.error("No messages found in conversation")
                return None
        except Exception as e:
            logger.error(f"Error extracting messages: {e}")
            return None

    @staticmethod
    def get_message_source(msg: Union[Dict[str, Any], Any]) -> Optional[str]:
        """
        Extract source identifier from message.

        Args:
            msg: Message object or dict

        Returns:
            Optional[str]: Source name or None
        """
        if hasattr(msg, "name"):
            return msg.name
        elif isinstance(msg, dict) and "name" in msg:
            return msg.get("name")
        return None

    @staticmethod
    def get_message_content(msg: Union[Dict[str, Any], Any]) -> Optional[Any]:
        """
        Extract content from message.

        Args:
            msg: Message object or dict

        Returns:
            Optional[Any]: Message content or None
        """
        if hasattr(msg, "content"):
            return msg.content
        elif isinstance(msg, dict) and "content" in msg:
            return msg.get("content")
        return None


class ConversationOrchestrator:
    """Orchestrates data extraction and processing from conversations."""

    def __init__(self):
        self.extractor = MessageExtractor()

    def extract_displayable_data(self, conversation: Union[Dict[str, Any], Any]) -> Dict[str, List[Any]]:
        """
        Extract displayable data from conversation.

        This method processes messages and categorizes them by source,
        returning a structured dictionary for display.

        Args:
            conversation: Conversation object or dictionary

        Returns:
            Dict: Categorized displayable data by source

        Example:
            >>> result = orchestrator.extract_displayable_data(conversation)
            >>> print(result['Work Flows'])  # List of workflow outputs
        """
        displayable_data = {}
        messages = self.extractor.get_messages(conversation)

        if not messages:
            return {}

        for msg in messages:
            source = self.extractor.get_message_source(msg)

            if not source:
                continue

            # Map source to display name
            display_source = MessageExtractor.SOURCE_MAPPING.get(source)
            if not display_source:
                continue

            content = self.extractor.get_message_content(msg)
            if not self._is_valid_content(content):
                continue

            if display_source not in displayable_data:
                displayable_data[display_source] = []

            displayable_data[display_source].append(content)

        return displayable_data

    def extract_displayable_data_v2(
        self, conversation: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """
        Extract displayable data from conversation (v2 format).

        Processes conversations with explicit output fields.

        Args:
            conversation: Conversation dictionary with output fields

        Returns:
            Dict: Categorized displayable data
        """
        displayable_data = {}

        field_mapping = {
            "output_from_workflow": "Work Flows",
            "output_from_web": "Web Researcher",
            "output_from_internal": "Internal Search",
            "output_from_worker": "Worker"
        }

        for field, display_name in field_mapping.items():
            if field in conversation:
                displayable_data[display_name] = [conversation[field]]

        return displayable_data

    @staticmethod
    def _is_valid_content(content: Any) -> bool:
        """
        Validate message content.

        Args:
            content: Content to validate

        Returns:
            bool: True if content is valid
        """
        if isinstance(content, str):
            return bool(content.strip())
        elif isinstance(content, list) and content:
            return hasattr(content[0], "content") or isinstance(content[0], str)
        return False


class TokenCounter:
    """Manages token count extraction from conversations."""

    @staticmethod
    def extract_token_counts(
        conversation: Union[Dict[str, Any], Any]
    ) -> Dict[str, int]:
        """
        Extract token counts from conversation.

        Args:
            conversation: Conversation object or dict

        Returns:
            Dict: Token counts with keys 'prompt_tokens', 'completion_tokens'

        Raises:
            ValueError: If no valid token data found
        """
        messages = MessageExtractor.get_messages(conversation)

        if not messages:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        for msg in messages:
            # Try to get token counts from models_usage
            usage = TokenCounter._extract_usage_from_models(msg)
            if usage:
                return usage

        return {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def extract_token_counts_v2(
        conversation: Union[Dict[str, Any], Any]
    ) -> Dict[str, int]:
        """
        Extract token counts from conversation (v2 format with metadata).

        Args:
            conversation: Conversation object with model_extra metadata

        Returns:
            Dict: Aggregated token counts
        """
        messages = MessageExtractor.get_messages(conversation)

        if not messages:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        prompt_tokens = 0
        completion_tokens = 0

        for msg in messages:
            if hasattr(msg, "model_extra") and msg.model_extra:
                metadata = msg.model_extra.get("usage_metadata")
                if metadata:
                    prompt_tokens += metadata.get("input_tokens", 0)
                    completion_tokens += metadata.get("total_tokens", 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    @staticmethod
    def _extract_usage_from_models(msg: Any) -> Optional[Dict[str, int]]:
        """
        Extract token counts from models_usage attribute.

        Args:
            msg: Message object

        Returns:
            Optional[Dict]: Token counts or None
        """
        if not hasattr(msg, "models_usage") or msg.models_usage is None:
            return None

        try:
            return {
                "prompt_tokens": msg.models_usage.prompt_tokens,
                "completion_tokens": msg.models_usage.completion_tokens
            }
        except AttributeError:
            return None


class ConversationHistory:
    """Manages conversation history retrieval and formatting."""

    DEFAULT_HISTORY_LIMIT = 6

    def __init__(self, sql_tool: Any):
        """
        Initialize history manager.

        Args:
            sql_tool: Database tool for retrieving history
        """
        self.sql_tool = sql_tool

    def get_history(
        self,
        user_id: str,
        symbol: str,
        chat_id: Optional[str] = None,
        limit: int = DEFAULT_HISTORY_LIMIT
    ) -> List[Tuple[str, str]]:
        """
        Retrieve conversation history.

        Args:
            user_id: User identifier
            symbol: Stock symbol
            chat_id: Optional chat session ID
            limit: Maximum number of messages to retrieve (default: 6)

        Returns:
            List of (question, answer) tuples

        Example:
            >>> history = conversation_history.get_history(
            ...     user_id="user123",
            ...     symbol="AAPL",
            ...     limit=5
            ... )
            >>> for question, answer in history:
            ...     print(f"Q: {question}")
            ...     print(f"A: {answer}")
        """
        try:
            history_data = self.sql_tool.update_read_data_testing(
                symbol=symbol,
                user_id=user_id,
                chat_id=chat_id
            )

            if not history_data or not history_data[0]:
                return []

            # Get only the last N messages
            last_messages = history_data[0][-limit:]

            return self._format_message_pairs(last_messages)

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    @staticmethod
    def _format_message_pairs(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Format messages into question-answer pairs.

        Args:
            messages: List of message dictionaries

        Returns:
            List of (question, answer) tuples
        """
        history = []

        for i in range(0, len(messages), 2):
            question = messages[i].get("content", "")

            if i + 1 < len(messages):
                answer = messages[i + 1].get("content", "")
            else:
                # Unmatched question without answer
                answer = ""

            history.append((question, answer))

        return history


def create_orchestrator() -> ConversationOrchestrator:
    """
    Factory function to create conversation orchestrator.

    Returns:
        ConversationOrchestrator: Initialized orchestrator
    """
    return ConversationOrchestrator()


def create_history_manager(sql_tool: Any) -> ConversationHistory:
    """
    Factory function to create conversation history manager.

    Args:
        sql_tool: Database tool

    Returns:
        ConversationHistory: Initialized history manager
    """
    return ConversationHistory(sql_tool)
