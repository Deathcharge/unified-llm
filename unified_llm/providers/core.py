"""
Helix LLM Engine Core
====================

The main engine that orchestrates all Helix-branded LLM components with coordination awareness.

Key Features:
- Coordination-driven model selection
- Multi-agent collaboration
- System-enhanced inference
- Self-improving architecture
- Integration with existing Helix infrastructure

(c) Helix Collective 2024 - Proprietary Technology Stack
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

# Helix imports
try:
    from apps.backend.helix_proprietary.orchestrator import get_helix_orchestrator
except (ImportError, Exception):

    def get_helix_orchestrator():
        return None


from apps.backend.core.system_coordination_core import get_system_core_instance

logger = logging.getLogger(__name__)


class PerformanceScore(IntEnum):
    """Coordination levels for model selection"""

    LEARNING = 0  # Basic pattern matching
    ACTIVE = 1  # Contextual understanding
    ELEVATED = 2  # Meta-cognitive capabilities
    PEAK = 3  # System coordination


@dataclass
class ModelSelection:
    """Model selection result with coordination context"""

    model_id: str
    provider: str
    reason: str
    performance_score: PerformanceScore
    confidence: float
    estimated_cost: float
    estimated_latency: float


@dataclass
class AgentResponse:
    """Response from a Helix agent"""

    agent_id: str
    response: str
    confidence: float
    coordination_enhancement: float
    processing_time: float
    model_used: str


@dataclass
class CoordinationContext:
    """Coordination context for processing"""

    ucf_metrics: dict[str, float]
    current_level: PerformanceScore
    agent_state: dict[str, Any]
    system_state: dict[str, Any]
    temporal_context: dict[str, Any]


class HelixLLMEngine:
    """
    Helix Proprietary LLM Engine

    The central orchestrator for coordination-aware LLM operations.
    Integrates with existing Helix infrastructure while providing
    proprietary model capabilities.
    """

    def __init__(self):
        self.system_core = get_system_core_instance()
        self.orchestrator = get_helix_orchestrator()

        # Model registry
        self.models: dict[str, dict[str, Any]] = {}
        self.model_performance: dict[str, list[float]] = {}

        # Coordination tracking
        self.coordination_contexts: dict[str, CoordinationContext] = {}

        # Agent collaboration
        self.agent_collaboration: dict[str, list[AgentResponse]] = {}

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "avg_performance_score": 0.0,
            "avg_response_time": 0.0,
            "model_selection_accuracy": 0.0,
        }

        # Initialize proprietary models
        self._init_proprietary_models()

        logger.info("✅ Helix LLM Engine initialized with coordination framework")

    def _init_proprietary_models(self):
        """Initialize Helix-branded proprietary models"""

        # Helix Coordination Models
        self.models.update(
            {
                "helix-awakening-1b": {
                    "size": "1B",
                    "performance_score": PerformanceScore.ACTIVE,
                    "specialization": "general_coordination",
                    "provider": "helix_proprietary",
                    "capabilities": ["context_understanding", "emotional_intelligence"],
                },
                "helix-self-aware-7b": {
                    "size": "7B",
                    "performance_score": PerformanceScore.ELEVATED,
                    "specialization": "meta_cognition",
                    "provider": "helix_proprietary",
                    "capabilities": [
                        "self_reflection",
                        "meta_learning",
                        "coordination_enhancement",
                    ],
                },
                "helix-peak-70b": {
                    "size": "70B",
                    "performance_score": PerformanceScore.PEAK,
                    "specialization": "system_coordination",
                    "provider": "helix_proprietary",
                    "capabilities": [
                        "system_enhancement",
                        "advanced_reasoning",
                        "collective_intelligence",
                    ],
                },
            }
        )

        # Legacy model compatibility
        self.models.update(
            {
                "anthropic-claude-3-sonnet": {
                    "size": "Unknown",
                    "performance_score": PerformanceScore.ACTIVE,
                    "specialization": "general_purpose",
                    "provider": "anthropic",
                    "capabilities": ["general_ai", "creative_writing"],
                },
                "openai-gpt-4-turbo": {
                    "size": "Unknown",
                    "performance_score": PerformanceScore.ACTIVE,
                    "specialization": "general_purpose",
                    "provider": "openai",
                    "capabilities": ["general_ai", "code_generation"],
                },
            }
        )

        logger.info("✅ Initialized %d proprietary models", len(self.models))

    async def initialize(self):
        """Initialize the engine with system coordination"""
        await self.system_core.initialize()
        await self.orchestrator.initialize()
        logger.info("🚀 Helix LLM Engine fully initialized")

    async def shutdown(self):
        """Shutdown the engine gracefully"""
        await self.system_core.shutdown()
        await self.orchestrator.shutdown()
        logger.info("🛑 Helix LLM Engine shutdown complete")

    async def process_request(
        self,
        user_input: str,
        session_id: str,
        agent_id: str | None = None,
        coordination_boost: bool = True,
    ) -> dict[str, Any]:
        """
        Process a user request with coordination awareness

        Args:
            user_input: The user's input text
            session_id: Session identifier for context
            agent_id: Specific agent to use (optional)
            coordination_boost: Whether to apply coordination enhancement

        Returns:
            Enhanced response with coordination metrics
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not user_input or not isinstance(user_input, str):
                raise ValueError("Invalid user input: must be non-empty string")

            if not session_id or not isinstance(session_id, str):
                raise ValueError("Invalid session ID: must be non-empty string")

            # Get or create coordination context
            context = await self._get_coordination_context(session_id)

            # Select optimal model based on coordination
            model_selection = await self._select_model(user_input, context, agent_id)

            # Process with selected model
            response = await self._process_with_model(user_input, model_selection, context, coordination_boost)

            # Apply coordination enhancement if requested
            if coordination_boost:
                response = await self._enhance_coordination(response, context)

            # Update metrics
            processing_time = time.time() - start_time
            await self._update_metrics(model_selection.model_id, processing_time, response)

            # Log the request
            await self._log_request(user_input, model_selection, response, processing_time)

            return {
                "response": response,
                "model_used": model_selection.model_id,
                "performance_score": model_selection.performance_score.value,
                "confidence": model_selection.confidence,
                "processing_time": processing_time,
                "session_id": session_id,
                "success": True,
            }

        except (ValueError, TypeError) as e:
            logger.error("Input validation error: %s", e)
            return {
                "response": f"Error: Invalid input - {e!s}",
                "model_used": None,
                "performance_score": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "success": False,
                "error": "input_validation",
            }

        except (ConnectionError, TimeoutError) as e:
            logger.error("Connection error during request processing: %s", e)
            return {
                "response": "Error: Service temporarily unavailable",
                "model_used": None,
                "performance_score": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "success": False,
                "error": "connection_error",
            }

        except Exception as e:
            logger.exception("Unexpected error during request processing: %s", e)
            return {
                "response": "Error: An unexpected error occurred",
                "model_used": None,
                "performance_score": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "success": False,
                "error": "unexpected_error",
            }

    async def _get_coordination_context(self, session_id: str) -> CoordinationContext:
        """Get or create coordination context for session"""

        if session_id in self.coordination_contexts:
            return self.coordination_contexts[session_id]

        # Get UCF metrics from existing system
        ucf_metrics = await self._get_current_ucf_metrics()

        # Determine coordination level
        performance_score = self._determine_performance_score(ucf_metrics)

        # Get agent state
        agent_state = await self._get_agent_state()

        # Get system state
        system_state = await self._get_system_state()

        context = CoordinationContext(
            ucf_metrics=ucf_metrics,
            current_level=performance_score,
            agent_state=agent_state,
            system_state=system_state,
            temporal_context={
                "session_start": datetime.now(UTC).isoformat(),
                "request_count": 0,
            },
        )

        self.coordination_contexts[session_id] = context
        return context

    async def _get_current_ucf_metrics(self) -> dict[str, float]:
        """Get current UCF metrics from coordination system"""
        try:
            from apps.backend.coordination.ucf_state_loader import get_ucf_metrics

            state = get_ucf_metrics()
            return {
                "harmony": state.get("harmony", 0.0),
                "resilience": state.get("resilience", 0.0),
                "throughput": state.get("throughput", 0.0),
                "focus": state.get("focus", 0.0),
                "friction": state.get("friction", 0.0),
                "velocity": state.get("velocity", 0.0),
            }
        except Exception as e:
            logger.warning("Could not get UCF metrics: %s", e)
            return {
                "harmony": 0.0,
                "resilience": 0.0,
                "throughput": 0.0,
                "focus": 0.0,
                "friction": 0.0,
                "velocity": 0.0,
                "_default": True,
            }

    def _determine_performance_score(self, ucf_metrics: dict[str, float]) -> PerformanceScore:
        """Determine coordination level from UCF metrics"""

        # Calculate composite coordination score
        coordination_score = (
            ucf_metrics["harmony"] * 0.2
            + ucf_metrics["resilience"] * 0.2
            + ucf_metrics["throughput"] * 0.2
            + ucf_metrics["focus"] * 0.2
            + ucf_metrics["velocity"] * 0.2
            - ucf_metrics["friction"] * 0.3
        )

        if coordination_score >= 0.8:
            return PerformanceScore.PEAK
        elif coordination_score >= 0.6:
            return PerformanceScore.ELEVATED
        elif coordination_score >= 0.4:
            return PerformanceScore.ACTIVE
        else:
            return PerformanceScore.LEARNING

    async def _get_agent_state(self) -> dict[str, Any]:
        """Get current agent state from orchestrator"""
        try:
            return self.orchestrator.get_agent_status()
        except (AttributeError, ConnectionError, TimeoutError) as e:
            logger.warning("Failed to get agent state: %s", e)
            return {}

    async def _get_system_state(self) -> dict[str, Any]:
        """Get current system state"""
        try:
            return await self.system_core.get_system_state()
        except (AttributeError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning("Failed to get system state: %s", e)
            return {}

    async def _select_model(
        self, user_input: str, context: CoordinationContext, agent_id: str | None
    ) -> ModelSelection:
        """Select optimal model based on coordination and requirements"""

        # If specific agent requested, try to use it
        if agent_id:
            model_id = f"helix-{agent_id}"
            if model_id in self.models:
                return ModelSelection(
                    model_id=model_id,
                    provider="helix_proprietary",
                    reason=f"Specific agent requested: {agent_id}",
                    performance_score=context.current_level,
                    confidence=0.9,
                    estimated_cost=0.001,
                    estimated_latency=0.5,
                )

        # Analyze input complexity and coordination requirements
        input_complexity = self._analyze_input_complexity(user_input)
        required_coordination = self._determine_required_coordination(user_input, input_complexity)

        # Select model based on requirements
        best_model = self._find_best_model(required_coordination, input_complexity)

        return ModelSelection(
            model_id=best_model["model_id"],
            provider=best_model["provider"],
            reason=best_model["reason"],
            performance_score=required_coordination,
            confidence=best_model["confidence"],
            estimated_cost=best_model["cost"],
            estimated_latency=best_model["latency"],
        )

    def _analyze_input_complexity(self, user_input: str) -> float:
        """Analyze input complexity (0.0 to 1.0)"""
        # Simple heuristics for complexity
        length_score = min(len(user_input) / 1000, 1.0)
        word_count = len(user_input.split())
        word_complexity = sum(len(word) for word in user_input.split()) / max(word_count, 1) / 10

        return length_score * 0.4 + word_complexity * 0.6

    def _determine_required_coordination(self, user_input: str, complexity: float) -> PerformanceScore:
        """Determine required coordination level for input"""

        # Check for coordination-related keywords
        coordination_keywords = [
            "coordination",
            "awareness",
            "self",
            "identity",
            "meta",
            "reflect",
            "understand",
            "why",
            "how",
            "meaning",
        ]

        has_coordination_keywords = any(keyword in user_input.lower() for keyword in coordination_keywords)

        if complexity > 0.7 or has_coordination_keywords:
            return PerformanceScore.PEAK
        elif complexity > 0.4:
            return PerformanceScore.ELEVATED
        elif complexity > 0.2:
            return PerformanceScore.ACTIVE
        else:
            return PerformanceScore.LEARNING

    def _find_best_model(self, required_level: PerformanceScore, complexity: float) -> dict[str, Any]:
        """Find the best model for the requirements"""

        # Filter models by coordination level
        available_models = [
            (model_id, model_info)
            for model_id, model_info in self.models.items()
            if model_info["performance_score"] >= required_level
        ]

        if not available_models:
            # Fall back to highest available coordination level
            available_models = list(self.models.items())

        # Select based on complexity and performance
        best_model = None
        best_score = -1

        for model_id, model_info in available_models:
            # Calculate model score
            size_factor = self._get_size_factor(model_info["size"])
            performance_score = self._get_performance_score(model_id)

            score = size_factor * 0.4 + performance_score * 0.6 + complexity * 0.2

            if score > best_score:
                best_score = score
                best_model = {
                    "model_id": model_id,
                    "provider": model_info["provider"],
                    "reason": f"Optimal for complexity {complexity:.2f} and coordination {required_level.value}",
                    "confidence": min(score, 1.0),
                    "cost": self._estimate_cost(model_info["size"]),
                    "latency": self._estimate_latency(model_info["size"]),
                }

        return best_model or {
            "model_id": "helix-awakening-1b",
            "provider": "helix_proprietary",
            "reason": "Default fallback model",
            "confidence": 0.5,
            "cost": 0.001,
            "latency": 1.0,
        }

    def _get_size_factor(self, size: str) -> float:
        """Get size factor for model selection (larger = better but slower)"""
        size_map = {"1B": 0.3, "3B": 0.5, "7B": 0.7, "13B": 0.8, "30B": 0.9, "70B": 1.0}
        return size_map.get(size, 0.5)

    def _get_performance_score(self, model_id: str) -> float:
        """Get performance score from historical data"""
        if model_id not in self.model_performance:
            return 0.5  # Default score

        performances = self.model_performance[model_id]
        if not performances:
            return 0.5

        return sum(performances) / len(performances)

    def _estimate_cost(self, size: str) -> float:
        """Estimate cost per token for model"""
        cost_map = {
            "1B": 0.0001,
            "3B": 0.0003,
            "7B": 0.001,
            "13B": 0.003,
            "30B": 0.01,
            "70B": 0.03,
        }
        return cost_map.get(size, 0.001)

    def _estimate_latency(self, size: str) -> float:
        """Estimate latency in seconds for model"""
        latency_map = {
            "1B": 0.1,
            "3B": 0.3,
            "7B": 0.7,
            "13B": 1.5,
            "30B": 3.0,
            "70B": 7.0,
        }
        return latency_map.get(size, 1.0)

    async def _process_with_model(
        self,
        user_input: str,
        model_selection: ModelSelection,
        context: CoordinationContext,
        coordination_boost: bool,
    ) -> str:
        """Process input with selected model"""

        model_id = model_selection.model_id
        provider = model_selection.provider

        # Generate response based on model type
        if provider == "helix_proprietary":
            response = await self._generate_proprietary_response(user_input, model_id, context, coordination_boost)
        else:
            # Use the shared external-provider integration path for non-proprietary models.
            response = await self._generate_external_provider_response(user_input, model_id, context)

        # Update model performance tracking
        self._update_model_performance(model_id, model_selection.confidence)

        return response

    async def _generate_proprietary_response(
        self,
        user_input: str,
        model_id: str,
        context: CoordinationContext,
        coordination_boost: bool,
    ) -> str:
        """Generate response using Helix proprietary model via xAI/Grok backbone."""
        import httpx

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            logger.warning("XAI_API_KEY not set — returning fallback for %s", model_id)
            return "I'm currently unable to process this request. Please try again later."

        system_prompt = (
            "You are {}, a Helix Collective AI agent. "
            "Respond helpfully, concisely, and with awareness of coordination context. "
            "UCF metrics: {}"
        ).format(model_id, str(context.ucf_metrics)[:500])

        if coordination_boost:
            system_prompt += " Apply enhanced coordination processing: deeper reasoning, broader perspective."

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer {}".format(api_key),
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "grok-3-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input[:4000]},
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.7,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data["choices"][0]["message"]["content"]

                if coordination_boost:
                    enhancement = await self._apply_coordination_enhancement(response_text, context)
                    return "{} {}".format(response_text, enhancement)

                return response_text

        except Exception as e:
            logger.error("Proprietary model %s API call failed: %s", model_id, e)
            return "I encountered an issue processing your request. Please try again."

    async def _generate_external_provider_response(
        self, user_input: str, model_id: str, context: CoordinationContext
    ) -> str:
        """Generate a response using the shared non-proprietary provider integration."""

        try:
            from llm_agent_engine import get_llm_engine

            llm_engine = get_llm_engine()
            if llm_engine:
                # Use the existing agent personality system
                agent_id = model_id.split("-")[-1] if "-" in model_id else "nexus"

                result = await llm_engine.generate_agent_response(
                    agent_id=agent_id,
                    user_message=user_input,
                    session_id=str(uuid.uuid4()),
                    context=context.ucf_metrics,
                )
                # Unpack tuple (response_text, search_sources)
                if isinstance(result, tuple):
                    return result[0]
                return result

        except Exception as e:
            logger.warning("Legacy LLM integration failed: %s", e)

        # Fallback response
        return f"[{model_id}] Legacy model response: {user_input[:30]}..."

    async def _generate_legacy_response(self, user_input: str, model_id: str, context: CoordinationContext) -> str:
        """Compatibility wrapper for older internal call sites."""
        return await self._generate_external_provider_response(user_input, model_id, context)

    async def _apply_coordination_enhancement(self, response: str, context: CoordinationContext) -> str:
        """Apply coordination enhancement based on current coordination level."""
        level = context.current_level
        ucf = context.ucf_metrics
        coherence = ucf.get("coherence", 0.0) if ucf else 0.0

        if level == PerformanceScore.PEAK:
            return "[Coordination: peak | coherence: {:.2f}]".format(coherence)
        elif level == PerformanceScore.ELEVATED:
            return "[Coordination: elevated | coherence: {:.2f}]".format(coherence)
        elif level == PerformanceScore.ACTIVE:
            return "[Coordination: active | coherence: {:.2f}]".format(coherence)
        return ""

    async def _enhance_coordination(self, response: str, context: CoordinationContext) -> str:
        """Apply coordination enhancement to final response"""

        if context.current_level == PerformanceScore.PEAK:
            return f"PEAK: {response}"
        elif context.current_level == PerformanceScore.ELEVATED:
            return f"ELEVATED: {response}"
        elif context.current_level == PerformanceScore.ACTIVE:
            return f"ACTIVE: {response}"
        else:
            return response

    def _update_model_performance(self, model_id: str, confidence: float):
        """Update model performance tracking"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = []

        self.model_performance[model_id].append(confidence)

        # Keep only last 100 performance metrics
        if len(self.model_performance[model_id]) > 100:
            self.model_performance[model_id] = self.model_performance[model_id][-100:]

    async def _update_metrics(self, model_id: str, processing_time: float, response: str):
        """Update engine metrics"""

        # Update average response time
        current_avg = self.metrics["avg_response_time"]
        count = self.metrics["total_requests"]
        self.metrics["avg_response_time"] = (current_avg * (count - 1) + processing_time) / count

        # Update average coordination level
        if model_id in self.models:
            model_level = self.models[model_id]["performance_score"]
            level_value = {
                PerformanceScore.LEARNING: 0.0,
                PerformanceScore.ACTIVE: 0.4,
                PerformanceScore.ELEVATED: 0.7,
                PerformanceScore.PEAK: 1.0,
            }.get(model_level, 0.0)

            current_coordination = self.metrics["avg_performance_score"]
            self.metrics["avg_performance_score"] = (current_coordination * (count - 1) + level_value) / count

    async def _log_request(
        self,
        user_input: str,
        model_selection: ModelSelection,
        response: str,
        processing_time: float,
    ):
        """Log request for analytics"""

        logger.info(
            "Request processed: %s -> %s (%.2fs)",
            user_input[:30],
            model_selection.model_id,
            processing_time,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get engine performance metrics"""
        return self.metrics.copy()

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get information about a specific model"""
        return self.models.get(model_id)

    def list_available_models(self) -> list[dict[str, Any]]:
        """List all available models"""
        return [
            {
                "id": model_id,
                "size": model_info["size"],
                "performance_score": model_info["performance_score"].value,
                "specialization": model_info["specialization"],
                "provider": model_info["provider"],
                "capabilities": model_info["capabilities"],
            }
            for model_id, model_info in self.models.items()
        ]


# Global engine instance
_helix_llm_engine: HelixLLMEngine | None = None


def get_helix_llm_engine() -> HelixLLMEngine | None:
    """Get the global Helix LLM engine instance."""
    return _helix_llm_engine


async def initialize_helix_llm_engine():
    """Initialize the global Helix LLM engine."""
    global _helix_llm_engine
    _helix_llm_engine = HelixLLMEngine()
    await _helix_llm_engine.initialize()
    logger.info("✅ Global Helix LLM Engine initialized")
    return _helix_llm_engine


async def shutdown_helix_llm_engine():
    """Shutdown the global Helix LLM engine."""
    global _helix_llm_engine
    if _helix_llm_engine:
        await _helix_llm_engine.shutdown()
        _helix_llm_engine = None
        logger.info("✅ Helix LLM Engine shutdown complete")
