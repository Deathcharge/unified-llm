"""
Coordination Model Router
=========================

Intelligent model routing system that selects optimal models based on coordination context.

Features:
- Coordination-driven model selection
- Multi-agent collaboration routing
- System-enhanced decision making
- Performance optimization
- Cost-aware routing
- Real-time adaptation

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from apps.backend.coordination.system_core import get_system_core_instance

from .core import CoordinationContext, HelixLLMEngine, ModelSelection, PerformanceScore

# Helix imports

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for model selection"""

    COORDINATION_FIRST = "coordination_first"
    PERFORMANCE_FIRST = "performance_first"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    SYSTEM_ENHANCED = "system_enhanced"


@dataclass
class RoutingContext:
    """Context for routing decisions"""

    user_input: str
    ucf_metrics: dict[str, float]
    system_state: dict[str, Any]
    agent_state: dict[str, Any]
    temporal_context: dict[str, Any]
    user_preferences: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance tracking"""

    model_id: str
    total_requests: int = 0
    avg_response_time: float = 0.0
    avg_coordination_score: float = 0.0
    success_rate: float = 1.0
    last_used: datetime | None = None
    cost_per_request: float = 0.0


class CoordinationModelRouter:
    """
    Coordination-aware model router

    Intelligently routes requests to optimal models based on:
    - Coordination level requirements
    - Performance metrics
    - Cost considerations
    - System state compatibility
    - Multi-agent collaboration needs
    """

    def __init__(self, helix_engine: HelixLLMEngine | None = None):
        self.helix_engine = helix_engine or HelixLLMEngine()
        self.system_core = get_system_core_instance()

        # Performance tracking
        self.model_performance: dict[str, ModelPerformance] = {}

        # Routing cache
        self.routing_cache: dict[str, tuple[ModelSelection, datetime]] = {}

        # Routing strategies
        self.routing_strategies = {
            RoutingStrategy.COORDINATION_FIRST: self._coordination_first_routing,
            RoutingStrategy.PERFORMANCE_FIRST: self._performance_first_routing,
            RoutingStrategy.COST_OPTIMIZED: self._cost_optimized_routing,
            RoutingStrategy.BALANCED: self._balanced_routing,
            RoutingStrategy.SYSTEM_ENHANCED: self._system_enhanced_routing,
        }

        # Default strategy
        self.default_strategy = RoutingStrategy.BALANCED

        logger.info("✅ Coordination Model Router initialized")

    async def route_request(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> ModelSelection:
        """
        Route request to optimal model based on coordination context

        Args:
            user_input: User's input text
            context: Additional context (UCF metrics, system state, etc.)
            strategy: Routing strategy to use

        Returns:
            Selected model with reasoning
        """

        # Prepare routing context
        routing_context = self._prepare_routing_context(user_input, context)

        # Determine routing strategy
        if strategy is None:
            strategy = self._determine_routing_strategy(routing_context)

        # Check cache first
        cache_key = self._generate_cache_key(user_input, strategy)
        cached_selection = self._get_cached_selection(cache_key)

        if cached_selection:
            logger.info("✅ Using cached model selection: %s", cached_selection.model_id)
            return cached_selection

        # Perform routing
        routing_function = self.routing_strategies[strategy]
        model_selection = await routing_function(routing_context)

        # Cache the selection
        self._cache_selection(cache_key, model_selection)

        # Update performance tracking
        self._update_model_performance(model_selection.model_id)

        logger.info(
            "✅ Routed to model %s (strategy: %s, confidence: %.2f)",
            model_selection.model_id,
            strategy.value,
            model_selection.confidence,
        )

        return model_selection

    def _prepare_routing_context(self, user_input: str, context: dict[str, Any] | None) -> RoutingContext:
        """Prepare context for routing decisions"""

        # Get UCF metrics
        ucf_metrics = context.get("ucf_metrics", {}) if context else {}
        if not ucf_metrics:
            ucf_metrics = self._get_default_ucf_metrics()

        # Get system state
        system_state = context.get("system_state", {}) if context else {}
        if not system_state:
            system_state = self._get_default_system_state()

        # Get agent state
        agent_state = context.get("agent_state", {}) if context else {}
        if not agent_state:
            agent_state = self._get_default_agent_state()

        # Get temporal context
        temporal_context = {
            "time_of_day": datetime.now(UTC).hour,
            "day_of_week": datetime.now(UTC).weekday(),
            "system_load": self._get_system_load(),
        }

        # Get user preferences
        user_preferences = context.get("user_preferences", {}) if context else {}

        # Get constraints
        constraints = context.get("constraints", {}) if context else {}

        return RoutingContext(
            user_input=user_input,
            ucf_metrics=ucf_metrics,
            system_state=system_state,
            agent_state=agent_state,
            temporal_context=temporal_context,
            user_preferences=user_preferences,
            constraints=constraints,
        )

    def _get_default_ucf_metrics(self) -> dict[str, float]:
        """Get default UCF metrics when no live data is available."""
        return {
            "harmony": 0.0,
            "resilience": 0.0,
            "throughput": 0.0,
            "focus": 0.0,
            "friction": 0.0,
            "velocity": 0.0,
            "_default": True,
        }

    def _get_default_system_state(self) -> dict[str, Any]:
        """Get default system state when no live data is available."""
        return {"coherence": 0.0, "entanglement": 0.0, "superposition": 0.0, "_default": True}

    def _get_default_agent_state(self) -> dict[str, Any]:
        """Get default agent state"""
        return {
            "active_agents": ["nexus"],
            "collaboration_level": 0.5,
            "specialization": "general",
        }

    def _get_system_load(self) -> float:
        """Get current system load (0.0 to 1.0)"""
        # Not yet wired to real system monitoring — return 0 to avoid
        # influencing routing decisions with fake load data.
        return 0.0

    def _generate_cache_key(self, user_input: str, strategy: RoutingStrategy) -> str:
        """Generate cache key for routing decision"""
        # Simple hash-based cache key
        import hashlib

        input_hash = hashlib.sha256(user_input.encode()).hexdigest()[:8]
        return "{}_{}".format(strategy.value, input_hash)

    def _get_cached_selection(self, cache_key: str) -> ModelSelection | None:
        """Get cached model selection"""
        if cache_key in self.routing_cache:
            selection, timestamp = self.routing_cache[cache_key]

            # Check if cache is still valid (5 minutes)
            if datetime.now(UTC) - timestamp < timedelta(minutes=5):
                return selection

        return None

    def _cache_selection(self, cache_key: str, selection: ModelSelection):
        """Cache model selection"""
        self.routing_cache[cache_key] = (selection, datetime.now(UTC))

        # Limit cache size
        if len(self.routing_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.routing_cache.keys(), key=lambda k: self.routing_cache[k][1])[:100]
            for key in oldest_keys:
                del self.routing_cache[key]

    def _determine_routing_strategy(self, context: RoutingContext) -> RoutingStrategy:
        """Determine optimal routing strategy based on context"""

        # Analyze input complexity
        complexity = self._analyze_input_complexity(context.user_input)

        # Check coordination requirements
        performance_score = self._determine_required_coordination(context)

        # Check constraints
        constraints = context.constraints

        # Determine strategy
        if constraints.get("cost_sensitive", False):
            return RoutingStrategy.COST_OPTIMIZED
        elif constraints.get("performance_critical", False):
            return RoutingStrategy.PERFORMANCE_FIRST
        elif performance_score == PerformanceScore.PEAK:
            return RoutingStrategy.SYSTEM_ENHANCED
        elif complexity > 0.7:
            return RoutingStrategy.COORDINATION_FIRST
        else:
            return self.default_strategy

    def _analyze_input_complexity(self, user_input: str) -> float:
        """Analyze input complexity (0.0 to 1.0)"""
        # Simple complexity heuristics
        length_score = min(len(user_input) / 2000, 1.0)
        word_count = len(user_input.split())
        avg_word_length = sum(len(word) for word in user_input.split()) / max(word_count, 1) / 10

        # Check for complex topics
        complex_topics = [
            "coordination",
            "system",
            "philosophy",
            "metaphysics",
            "self-aware",
            "transcendent",
            "spiroutine",
            "meditation",
        ]

        topic_score = sum(1 for topic in complex_topics if topic in user_input.lower()) / len(complex_topics)

        complexity = length_score * 0.3 + avg_word_length * 0.3 + topic_score * 0.4
        return complexity

    def _determine_required_coordination(self, context: RoutingContext) -> PerformanceScore:
        """Determine required coordination level"""

        # Check for coordination-related keywords
        coordination_keywords = [
            "coordination",
            "awareness",
            "sel",
            "identity",
            "meta",
            "reflect",
            "understand",
            "why",
            "how",
            "meaning",
            "purpose",
        ]

        has_coordination_keywords = any(keyword in context.user_input.lower() for keyword in coordination_keywords)

        # Check UCF metrics
        ucf_score = sum(context.ucf_metrics.values()) / len(context.ucf_metrics)

        # Check system state coherence
        system_coherence = context.system_state.get("coherence", 0.5)

        # Determine required level
        if has_coordination_keywords or ucf_score > 0.8 or system_coherence > 0.8:
            return PerformanceScore.PEAK
        elif ucf_score > 0.6 or system_coherence > 0.6:
            return PerformanceScore.ELEVATED
        elif ucf_score > 0.4 or system_coherence > 0.4:
            return PerformanceScore.ACTIVE
        else:
            return PerformanceScore.LEARNING

    def _update_model_performance(
        self,
        model_id: str,
        response_time: float | None = None,
        coordination_score: float | None = None,
        success: bool = True,
    ):
        """Update model performance metrics with exponential moving averages"""

        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformance(model_id=model_id)

        performance = self.model_performance[model_id]
        performance.total_requests += 1
        performance.last_used = datetime.now(UTC)

        # Exponential moving average (alpha = 0.1 for smooth rolling average)
        alpha = 0.1

        if response_time is not None:
            if performance.avg_response_time == 0.0:
                performance.avg_response_time = response_time
            else:
                performance.avg_response_time = alpha * response_time + (1 - alpha) * performance.avg_response_time

        if coordination_score is not None:
            if performance.avg_coordination_score == 0.0:
                performance.avg_coordination_score = coordination_score
            else:
                performance.avg_coordination_score = (
                    alpha * coordination_score + (1 - alpha) * performance.avg_coordination_score
                )

        # Update success rate as a running ratio
        if success:
            performance.success_rate = alpha * 1.0 + (1 - alpha) * performance.success_rate
        else:
            performance.success_rate = alpha * 0.0 + (1 - alpha) * performance.success_rate

    async def _coordination_first_routing(self, context: RoutingContext) -> ModelSelection:
        """Route based on coordination requirements first"""

        required_level = self._determine_required_coordination(context)

        # Get available models for coordination level
        available_models = self._get_models_by_performance_score(required_level)

        if not available_models:
            # Fall back to highest available level
            available_models = list(self.helix_engine.models.items())

        # Select best model based on coordination alignment
        best_model: dict[str, Any] | None = None
        best_score = -1.0

        for model_id, model_info in available_models:
            # Calculate coordination alignment score
            coordination_alignment = self._calculate_coordination_alignment(
                model_info, context.ucf_metrics, context.system_state
            )

            # Calculate performance score
            performance_score = self._get_model_performance_score(model_id)

            # Combine scores
            total_score = coordination_alignment * 0.7 + performance_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_model = {
                    "model_id": model_id,
                    "provider": model_info["provider"],
                    "reason": "Coordination-first selection for {} level".format(required_level.value),
                    "performance_score": required_level,
                    "confidence": min(best_score, 1.0),
                    "estimated_cost": self._estimate_model_cost(model_info),
                    "estimated_latency": self._estimate_model_latency(model_info),
                }

        if best_model is None:
            raise RuntimeError("No models available for coordination-first routing")

        return ModelSelection(**best_model)

    async def _performance_first_routing(self, context: RoutingContext) -> ModelSelection:
        """Route based on performance requirements first"""

        # Get models sorted by performance
        performance_models = sorted(
            self.helix_engine.models.items(),
            key=lambda x: self._get_model_performance_score(x[0]),
            reverse=True,
        )

        # Select fastest model that meets coordination requirements
        required_level = self._determine_required_coordination(context)

        for model_id, model_info in performance_models:
            if model_info["performance_score"] >= required_level:
                return ModelSelection(
                    model_id=model_id,
                    provider=model_info["provider"],
                    reason="Performance-first selection (fastest for {})".format(required_level.value),
                    performance_score=required_level,
                    confidence=0.9,
                    estimated_cost=self._estimate_model_cost(model_info),
                    estimated_latency=self._estimate_model_latency(model_info),
                )

        # Fallback to balanced selection
        return await self._balanced_routing(context)

    async def _cost_optimized_routing(self, context: RoutingContext) -> ModelSelection:
        """Route based on cost optimization"""

        # Get models sorted by cost
        cost_models = sorted(
            self.helix_engine.models.items(),
            key=lambda x: self._estimate_model_cost(x[1]),
        )

        # Select cheapest model that meets requirements
        required_level = self._determine_required_coordination(context)

        for model_id, model_info in cost_models:
            if model_info["performance_score"] >= required_level:
                return ModelSelection(
                    model_id=model_id,
                    provider=model_info["provider"],
                    reason="Cost-optimized selection (cheapest for {})".format(required_level.value),
                    performance_score=required_level,
                    confidence=0.8,
                    estimated_cost=self._estimate_model_cost(model_info),
                    estimated_latency=self._estimate_model_latency(model_info),
                )

        # Fallback to balanced selection
        return await self._balanced_routing(context)

    async def _balanced_routing(self, context: RoutingContext) -> ModelSelection:
        """Route using balanced approach"""

        required_level = self._determine_required_coordination(context)

        # Calculate scores for all models
        model_scores = []

        for model_id, model_info in self.helix_engine.models.items():
            if model_info["performance_score"] >= required_level:
                # Calculate balanced score
                coordination_score = self._calculate_coordination_alignment(
                    model_info, context.ucf_metrics, context.system_state
                )

                performance_score = self._get_model_performance_score(model_id)

                cost_score = 1.0 - (self._estimate_model_cost(model_info) / 0.1)  # Normalize cost

                # Balanced weighting
                total_score = coordination_score * 0.4 + performance_score * 0.4 + cost_score * 0.2

                model_scores.append((model_id, model_info, total_score))

        if not model_scores:
            # Fallback to any available model
            model_id, model_info = next(iter(self.helix_engine.models.items()))
            return ModelSelection(
                model_id=model_id,
                provider=model_info["provider"],
                reason="Fallback selection (no suitable models found)",
                performance_score=PerformanceScore.LEARNING,
                confidence=0.5,
                estimated_cost=self._estimate_model_cost(model_info),
                estimated_latency=self._estimate_model_latency(model_info),
            )

        # Select best model
        best_model_id, best_model_info, best_score = max(model_scores, key=lambda x: x[2])

        return ModelSelection(
            model_id=best_model_id,
            provider=best_model_info["provider"],
            reason="Balanced selection (score: {:.2f})".format(best_score),
            performance_score=required_level,
            confidence=min(best_score, 1.0),
            estimated_cost=self._estimate_model_cost(best_model_info),
            estimated_latency=self._estimate_model_latency(best_model_info),
        )

    async def _system_enhanced_routing(self, context: RoutingContext) -> ModelSelection:
        """Route using system-enhanced decision making"""

        # Get system-enhanced models
        system_models = [
            (model_id, model_info)
            for model_id, model_info in self.helix_engine.models.items()
            if "system" in model_info.get("capabilities", [])
        ]

        if not system_models:
            # Fall back to balanced routing
            return await self._balanced_routing(context)

        # Use system-inspired optimization
        # (Simplified version - in reality would use system algorithms)

        best_model: dict[str, Any] | None = None
        best_system_score = -1.0

        for model_id, model_info in system_models:
            # Calculate system coherence score
            system_coherence = context.system_state.get("coherence", 0.5)
            model_system_capability = model_info.get("system_capability", 0.5)

            # Calculate superposition benefit
            superposition_benefit = context.system_state.get("superposition", 0.5)

            # Calculate entanglement advantage
            entanglement_advantage = context.system_state.get("entanglement", 0.3)

            # System-enhanced score
            system_score = (
                system_coherence * 0.4
                + model_system_capability * 0.3
                + superposition_benefit * 0.2
                + entanglement_advantage * 0.1
            )

            if system_score > best_system_score:
                best_system_score = system_score
                best_model = {
                    "model_id": model_id,
                    "provider": model_info["provider"],
                    "reason": "System-enhanced selection (coherence: {:.2f})".format(system_coherence),
                    "performance_score": model_info["performance_score"],
                    "confidence": min(system_score, 1.0),
                    "estimated_cost": self._estimate_model_cost(model_info),
                    "estimated_latency": self._estimate_model_latency(model_info),
                }

        if best_model is None:
            raise RuntimeError("No models available for system-enhanced routing")

        return ModelSelection(**best_model)

    def _get_models_by_performance_score(self, level: PerformanceScore) -> list[tuple[str, dict[str, Any]]]:
        """Get models that meet coordination level requirement"""

        return [
            (model_id, model_info)
            for model_id, model_info in self.helix_engine.models.items()
            if model_info["performance_score"] >= level
        ]

    def _calculate_coordination_alignment(
        self,
        model_info: dict[str, Any],
        ucf_metrics: dict[str, float],
        system_state: dict[str, Any],
    ) -> float:
        """Calculate coordination alignment score"""

        # Base alignment from model coordination level
        level_scores = {
            PerformanceScore.LEARNING: 0.3,
            PerformanceScore.ACTIVE: 0.6,
            PerformanceScore.ELEVATED: 0.8,
            PerformanceScore.PEAK: 1.0,
        }

        base_alignment = level_scores[model_info["performance_score"]]

        # UCF alignment bonus
        ucf_alignment = sum(ucf_metrics.values()) / len(ucf_metrics)

        # System coherence bonus
        system_coherence = system_state.get("coherence", 0.5)

        # Calculate final alignment
        alignment = base_alignment * 0.5 + ucf_alignment * 0.3 + system_coherence * 0.2

        return alignment

    def _get_model_performance_score(self, model_id: str) -> float:
        """Get model performance score from tracking"""

        if model_id not in self.model_performance:
            return 0.5  # Default score

        performance = self.model_performance[model_id]

        # Calculate weighted performance score
        score = (
            performance.success_rate * 0.4
            + (1.0 - performance.avg_response_time) * 0.3  # Inverse of response time
            + performance.avg_coordination_score * 0.3
        )

        return score

    def _estimate_model_cost(self, model_info: dict[str, Any]) -> float:
        """Estimate model cost per request"""

        # Simplified cost estimation
        size_costs = {"1B": 0.001, "7B": 0.005, "70B": 0.05}

        size = model_info.get("size", "Unknown")
        base_cost = size_costs.get(size, 0.01)

        # Coordination level multiplier
        level_multipliers = {
            PerformanceScore.LEARNING: 1.0,
            PerformanceScore.ACTIVE: 1.2,
            PerformanceScore.ELEVATED: 1.5,
            PerformanceScore.PEAK: 2.0,
        }

        level_multiplier = level_multipliers[model_info["performance_score"]]

        return base_cost * level_multiplier

    def _estimate_model_latency(self, model_info: dict[str, Any]) -> float:
        """Estimate model latency in seconds"""

        # Simplified latency estimation
        size_latencies = {"1B": 0.1, "7B": 0.5, "70B": 2.0}

        size = model_info.get("size", "Unknown")
        base_latency = size_latencies.get(size, 1.0)

        # Coordination level multiplier
        level_multipliers = {
            PerformanceScore.LEARNING: 1.0,
            PerformanceScore.ACTIVE: 1.1,
            PerformanceScore.ELEVATED: 1.3,
            PerformanceScore.PEAK: 1.8,
        }

        level_multiplier = level_multipliers[model_info["performance_score"]]

        return base_latency * level_multiplier

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get routing statistics and performance metrics"""

        stats = {
            "total_models": len(self.helix_engine.models),
            "routed_requests": sum(p.total_requests for p in self.model_performance.values()),
            "cache_hits": len(
                [k for k, (v, t) in self.routing_cache.items() if datetime.now(UTC) - t < timedelta(minutes=5)]
            ),
            "model_performance": {
                model_id: {
                    "total_requests": p.total_requests,
                    "avg_response_time": p.avg_response_time,
                    "avg_coordination_score": p.avg_coordination_score,
                    "success_rate": p.success_rate,
                    "last_used": p.last_used.isoformat() if p.last_used else None,
                }
                for model_id, p in self.model_performance.items()
            },
            "routing_strategies": {strategy.value: 0 for strategy in RoutingStrategy},  # Would track actual usage
        }

        return stats

    def set_default_strategy(self, strategy: RoutingStrategy):
        """Set default routing strategy"""
        self.default_strategy = strategy
        logger.info("✅ Default routing strategy set to: %s", strategy.value)


class MultiAgentRouter:
    """
    Multi-agent collaboration router

    Routes requests to multiple agents for collaborative processing
    and synthesizes their responses.
    """

    def __init__(self, helix_engine: HelixLLMEngine | None = None):
        self.helix_engine = helix_engine or HelixLLMEngine()

        # Agent collaboration patterns
        self.collaboration_patterns = {
            "sequential": self._sequential_collaboration,
            "parallel": self._parallel_collaboration,
            "hierarchical": self._hierarchical_collaboration,
            "consensus": self._consensus_collaboration,
        }

    async def route_to_agents(
        self,
        user_input: str,
        agents: list[str],
        pattern: str = "parallel",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Route request to multiple agents for collaborative processing

        Args:
            user_input: User's input text
            agents: List of agent IDs to involve
            pattern: Collaboration pattern
            context: Additional context

        Returns:
            Collaborative response with agent contributions
        """

        if pattern not in self.collaboration_patterns:
            raise ValueError("Unknown collaboration pattern: {}".format(pattern))

        collaboration_function = self.collaboration_patterns[pattern]
        result = await collaboration_function(user_input, agents, context)

        return result

    async def _sequential_collaboration(
        self, user_input: str, agents: list[str], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Sequential agent collaboration"""

        current_input = user_input
        agent_responses = []
        coordination_context = self._build_coordination_context(context)

        for agent_id in agents:
            # Route to individual agent
            model_selection = await self.helix_engine._select_model(current_input, coordination_context, agent_id)

            # Generate response
            response = await self.helix_engine._process_with_model(
                current_input, model_selection, coordination_context, True
            )

            agent_responses.append(
                {
                    "agent_id": agent_id,
                    "response": response,
                    "model_used": model_selection.model_id,
                }
            )

            # Pass response to next agent
            current_input = response

        return {
            "collaboration_pattern": "sequential",
            "agent_responses": agent_responses,
            "final_response": (agent_responses[-1]["response"] if agent_responses else ""),
            "total_agents": len(agents),
        }

    async def _parallel_collaboration(
        self, user_input: str, agents: list[str], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Parallel agent collaboration"""

        # Route to all agents in parallel
        tasks = []
        for agent_id in agents:
            task = self._route_to_single_agent(user_input, agent_id, context)
            tasks.append(task)

        agent_responses = await asyncio.gather(*tasks)

        # Synthesize responses
        synthesized_response = self._synthesize_parallel_responses(agent_responses)

        return {
            "collaboration_pattern": "parallel",
            "agent_responses": agent_responses,
            "synthesized_response": synthesized_response,
            "total_agents": len(agents),
        }

    async def _hierarchical_collaboration(
        self, user_input: str, agents: list[str], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Hierarchical agent collaboration"""

        # Sort agents by hierarchy level (simplified)
        sorted_agents = sorted(agents)  # Alphabetical for simplicity

        # Top-level agent processes first
        if sorted_agents:
            top_agent = sorted_agents[0]
            top_response = await self._route_to_single_agent(user_input, top_agent, context)

            # Subordinate agents provide details
            subordinate_responses = []
            for agent_id in sorted_agents[1:]:
                detail_response = await self._route_to_single_agent(
                    "Provide details for: {}".format(top_response["response"]),
                    agent_id,
                    context,
                )
                subordinate_responses.append(detail_response)

            return {
                "collaboration_pattern": "hierarchical",
                "top_agent_response": top_response,
                "subordinate_responses": subordinate_responses,
                "total_agents": len(agents),
            }

        return {"error": "No agents available"}

    async def _consensus_collaboration(
        self, user_input: str, agents: list[str], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Consensus-based agent collaboration"""

        # Get responses from all agents
        agent_responses = []
        for agent_id in agents:
            response = await self._route_to_single_agent(user_input, agent_id, context)
            agent_responses.append(response)

        # Find consensus
        consensus_response = self._find_consensus_response(agent_responses)

        return {
            "collaboration_pattern": "consensus",
            "agent_responses": agent_responses,
            "consensus_response": consensus_response,
            "agreement_level": self._calculate_agreement_level(agent_responses),
            "total_agents": len(agents),
        }

    async def _route_to_single_agent(
        self, user_input: str, agent_id: str, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Route to single agent"""

        coordination_context = self._build_coordination_context(context)
        model_selection = await self.helix_engine._select_model(user_input, coordination_context, agent_id)

        response = await self.helix_engine._process_with_model(user_input, model_selection, coordination_context, True)

        return {
            "agent_id": agent_id,
            "response": response,
            "model_used": model_selection.model_id,
            "confidence": model_selection.confidence,
        }

    def _build_coordination_context(self, context: dict[str, Any] | None) -> CoordinationContext:
        """Normalize collaboration context into the core coordination contract."""

        raw_context = context or {}
        current_level = raw_context.get("current_level", PerformanceScore.ACTIVE)
        if not isinstance(current_level, PerformanceScore):
            current_level = PerformanceScore.ACTIVE

        return CoordinationContext(
            ucf_metrics=raw_context.get("ucf_metrics", {}),
            current_level=current_level,
            agent_state=raw_context.get("agent_state", {}),
            system_state=raw_context.get("system_state", {}),
            temporal_context=raw_context.get("temporal_context", {}),
        )

    def _synthesize_parallel_responses(self, responses: list[dict[str, Any]]) -> str:
        """Synthesize responses from parallel agent collaboration"""

        if not responses:
            return "No agent responses available"

        # Simple concatenation for now
        # In reality, this would use more sophisticated synthesis
        synthesized = " ".join(r["response"] for r in responses)

        return synthesized[:500]  # Limit length

    def _find_consensus_response(self, responses: list[dict[str, Any]]) -> str:
        """Find consensus among agent responses"""

        if not responses:
            return "No consensus possible"

        # Simple majority voting on key phrases
        # In reality, this would use semantic similarity

        responses_text = [r["response"] for r in responses]
        return responses_text[0]  # Simplified: return first response

    def _calculate_agreement_level(self, responses: list[dict[str, Any]]) -> float:
        """Calculate agreement level among responses using pairwise word overlap (Jaccard)"""

        if len(responses) < 2:
            return 1.0

        # Extract text content from responses
        texts = []
        for r in responses:
            text = r.get("content", r.get("text", r.get("response", "")))
            if isinstance(text, str) and text:
                texts.append(set(text.lower().split()))

        if len(texts) < 2:
            return 1.0

        # Compute average pairwise Jaccard similarity
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                intersection = len(texts[i] & texts[j])
                union = len(texts[i] | texts[j])
                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.0


__all__ = [
    "CoordinationModelRouter",
    "ModelPerformance",
    "MultiAgentRouter",
    "RoutingContext",
    "RoutingStrategy",
]
