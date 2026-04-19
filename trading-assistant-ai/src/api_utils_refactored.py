import logging
from collections import deque
from typing import List, Tuple, Optional
import re
import jwt

logger = logging.getLogger(__name__)


class FallbackAPIKeyManager:
    """
    Manages multiple API keys with automatic fallback and rotation.
    
    Supports rate limiting detection and key rotation for failed requests.
    
    Example:
        >>> manager = FallbackAPIKeyManager("primary_key", "backup_key")
        >>> key = manager.get_key()
        >>> if response.status_code == 429:
        ...     if manager.handle_rate_limit():
        ...         key = manager.get_key()  # Use next key
    """

    # HTTP status codes triggering key rotation
    RATE_LIMIT_STATUS = 429
    ERROR_STATUSES = {429, 401, 403}

    def __init__(self, primary_key: str, secondary_key: str):
        """
        Initialize key manager with primary and secondary keys.

        Args:
            primary_key: Primary API key
            secondary_key: Secondary/backup API key
        """
        self.keys = [primary_key, secondary_key]
        self.current_index = 0
        self.rotation_count = 0

        logger.info("APIKeyManager initialized with 2 keys")

    def get_key(self) -> str:
        """
        Get current active API key.

        Returns:
            str: Current API key
        """
        return self.keys[self.current_index]

    def get_current_key_index(self) -> int:
        """
        Get index of current key.

        Returns:
            int: Current key index
        """
        return self.current_index

    def rotate_key(self) -> None:
        """
        Rotate to next available key.

        Updates current_index and increments rotation counter.
        """
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.rotation_count += 1

        logger.warning(
            f"API key rotated to index {self.current_index} "
            f"(rotation #{self.rotation_count})"
        )

    def handle_rate_limit(self, status_code: int) -> bool:
        """
        Handle rate limiting or error response.

        Rotates key on rate limit (429) and other critical errors.

        Args:
            status_code: HTTP status code from API response

        Returns:
            bool: True if key was rotated, False otherwise

        Example:
            >>> if manager.handle_rate_limit(response.status_code):
            ...     # Try again with new key
        """
        if status_code == self.RATE_LIMIT_STATUS:
            logger.warning(f"Rate limit detected (status {status_code})")
            self.rotate_key()
            return True

        if status_code in self.ERROR_STATUSES:
            logger.warning(f"Error status {status_code} - attempting key rotation")
            self.rotate_key()
            return True

        return False

    def get_rotation_stats(self) -> dict:
        """
        Get rotation statistics.

        Returns:
            dict: Stats including current key index and total rotations
        """
        return {
            "current_key_index": self.current_index,
            "total_rotations": self.rotation_count,
            "current_key_length": len(self.get_key())
        }


class SentinelFilter:
    """
    Filters output stream by detecting termination patterns.
    
    Processes tokens one at a time and stops when any sentinel pattern
    is detected, enabling real-time filtering for streaming responses.
    
    Handles:
    - Direct token matches
    - Cross-token patterns (patterns spanning multiple tokens)
    - Word boundary detection
    - Configurable sentinel patterns
    
    Example:
        >>> filter = SentinelFilter(["TERMINATE", "END"])
        >>> for token in stream:
        ...     output, should_stop = filter.feed(token)
        ...     if output:
        ...         print(output, end='')
        ...     if should_stop:
        ...         break
        >>> final_text = filter.get_filtered_content()
    """

    # Buffer configuration
    MAX_BUFFER_SIZE = 100
    MIN_PATTERN_LENGTH = 2

    def __init__(self, sentinels: List[str]):
        """
        Initialize filter with sentinel patterns.

        Args:
            sentinels: List of termination patterns to detect
        """
        self.sentinels = [s.upper() for s in sentinels]
        self._compile_patterns()

        # Token and buffer management
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER_SIZE)
        self.buffer_text = ""
        self.raw_tokens: List[str] = []
        self.filtered_content = ""

        logger.info(f"SentinelFilter initialized with {len(self.sentinels)} sentinels")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.pattern_regexes = []

        for pattern in self.sentinels:
            # Create flexible pattern allowing spaces between characters
            flexible_pattern = (
                r"\b" + r"\s*".join(re.escape(char) for char in pattern) + r"\b"
            )
            self.pattern_regexes.append(
                re.compile(flexible_pattern, re.IGNORECASE)
            )

    def feed(self, token: str) -> Tuple[str, bool]:
        """
        Process a token and detect termination.

        Args:
            token: Token to process

        Returns:
            Tuple[str, bool]: (token_to_output, should_terminate)

        Example:
            >>> output, stop = filter.feed("This is")
            >>> print(output)  # "This is"
            >>> output, stop = filter.feed("TERMINATE")
            >>> print(stop)  # True
        """
        # Store raw token for analysis
        self.raw_tokens.append(token)

        # Fast path: direct token match
        if token.strip().upper() in self.sentinels:
            logger.debug(f"Sentinel detected: {token}")
            return "", True

        # Check if sentinel is embedded in word (e.g., "indeterminate")
        if self._is_embedded_sentinel(token):
            self.filtered_content += token
            return token, False

        # Add to buffer and check for cross-token patterns
        self._buffer.append(token)
        self.buffer_text = "".join(self._buffer)

        # Check regex patterns
        termination_index = self._check_patterns()
        if termination_index is not None:
            prefix = self.buffer_text[:termination_index]
            self.filtered_content += prefix
            self._buffer.clear()
            self.buffer_text = ""
            logger.debug("Pattern detected across tokens")
            return "", True

        # No sentinel found
        self.filtered_content += token
        return token, False

    def _is_embedded_sentinel(self, token: str) -> bool:
        """
        Check if sentinel is part of a larger word.

        Args:
            token: Token to check

        Returns:
            bool: True if sentinel is embedded in word
        """
        for pattern in self.sentinels:
            if pattern.upper() in token.upper():
                # Check if surrounded by word characters
                if re.search(r'\w' + re.escape(pattern) + r'\w', token, re.IGNORECASE):
                    return True
        return False

    def _check_patterns(self) -> Optional[int]:
        """
        Check buffer against compiled patterns.

        Returns:
            Optional[int]: Index where pattern starts, or None
        """
        for pattern_regex in self.pattern_regexes:
            match = pattern_regex.search(self.buffer_text)
            if match:
                return match.start()
        return None

    def get_filtered_content(self) -> str:
        """
        Get complete filtered output.

        Returns:
            str: All accepted tokens concatenated
        """
        return self.filtered_content

    def get_raw_tokens(self) -> List[str]:
        """
        Get raw token list.

        Returns:
            List[str]: Original tokens processed
        """
        return self.raw_tokens.copy()

    def reset(self) -> None:
        """Reset filter state for reuse."""
        self._buffer.clear()
        self.buffer_text = ""
        self.filtered_content = ""
        self.raw_tokens = []
        logger.debug("SentinelFilter reset")


class TokenDecoder:
    """
    Decodes and validates JWT tokens.
    
    Extracts claims from JWT tokens without verification (for client-side use).
    Supports HS256 algorithm.
    
    Example:
        >>> decoder = TokenDecoder()
        >>> user_id = decoder.decode(token, secret_key)
        >>> print(user_id)  # "user123"
    """

    def __init__(self):
        """Initialize token decoder."""
        self.algorithm = "HS256"

    def decode(self, token: str, secret_key: str) -> Optional[int]:
        """
        Decode JWT token and extract user ID.

        Args:
            token: JWT token string
            secret_key: Secret key for decoding

        Returns:
            Optional[int]: User ID from token 'sub' claim, or None if invalid

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(token, secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")

            if user_id is None:
                logger.warning("Token missing 'sub' claim")
                return None

            return int(user_id)

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting user_id to int: {e}")
            return None

    def decode_unsafe(self, token: str) -> Optional[dict]:
        """
        Decode JWT token without signature verification.

        **WARNING**: Use only for client-side parsing when verification
        cannot be done. Always verify tokens on the server.

        Args:
            token: JWT token string

        Returns:
            Optional[dict]: Token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            logger.info("Token decoded without verification (unsafe)")
            return payload

        except jwt.DecodeError as e:
            logger.error(f"Error decoding token: {e}")
            return None

    def get_user_id(self, token: str) -> Optional[int]:
        """
        Extract user ID from token (unsafe decode).

        Args:
            token: JWT token string

        Returns:
            Optional[int]: User ID or None
        """
        payload = self.decode_unsafe(token)
        if payload:
            try:
                return int(payload.get("sub"))
            except (ValueError, TypeError):
                logger.error("Cannot convert sub claim to int")
        return None
