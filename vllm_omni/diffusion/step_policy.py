# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Side-effect-free compatibility queries for scalar denoising-step consumers."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, cast


class StepExecutionKeyProvider(Protocol):
    def get_step_execution_key(self, step_idx: int) -> Hashable:
        """Return an immutable host key, without reading or changing ForwardContext.

        With all other execution conditions fixed, equal keys promise that any
        of those steps select the same execution choice. Owners
        must include every step-dependent choice and reuse their execution
        resolvers. This does not replace request-local timestep tensors.
        """
        ...


@dataclass(frozen=True)
class ExactStep:
    step_idx: int


def resolve_step_execution_key(owner: object | None, step_idx: int) -> Hashable:
    """An absent consumer is invariant; an undeclared consumer requires exact steps."""
    if type(step_idx) is not int or step_idx < 0:
        raise ValueError("Step execution keys require a non-negative integer denoise step.")
    if owner is None:
        return None
    if not callable(getattr(owner, "get_step_execution_key", None)):
        return ExactStep(step_idx)
    return cast(StepExecutionKeyProvider, owner).get_step_execution_key(step_idx)


@dataclass(frozen=True)
class DenoiseExecutionPolicy:
    """Immutable, forward-local choices indexed by live consumer identity.

    Equality is meaningful only within one model instance. Values are the
    actual execution choices, not representative step indices. Consumers must
    remain alive for this scope; this object is not a serializable config.
    """

    choices: tuple[tuple[int, Hashable], ...] = ()

    def with_consumer(self, consumer: object | None, step_idx: int) -> "DenoiseExecutionPolicy":
        choice = resolve_step_execution_key(consumer, step_idx)
        if consumer is None:
            return self
        return DenoiseExecutionPolicy((*self.choices, (id(consumer), choice)))

    def choice_for(self, consumer: object) -> Hashable:
        for identity, choice in self.choices:
            if identity == id(consumer):
                return choice
        raise RuntimeError("Active denoise execution policy has no choice for this consumer.")
