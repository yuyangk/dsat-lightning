"""Multi-GPU-friendly logger using loguru (only rank 0 when rank_zero_only=True)."""

from typing import Any, Optional

from loguru import logger as _loguru_logger
from lightning_utilities.core.rank_zero import rank_zero_only as _rank_zero_module


class RankedLogger:
    """A multi-GPU-friendly logger backed by loguru.

    When rank_zero_only=True, only the rank 0 process emits logs.
    When rank is not set (single process), logs are always emitted.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Any] = None,
    ) -> None:
        self._name = name
        self._rank_zero_only = rank_zero_only
        self._logger = _loguru_logger.bind(name=name)

    def _should_log(self) -> bool:
        if not self._rank_zero_only:
            return True
        rank = getattr(_rank_zero_module, "rank", None)
        if rank is None:
            return True  # single process, no distributed
        return rank == 0

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str = "", *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.exception(msg, *args, **kwargs)
