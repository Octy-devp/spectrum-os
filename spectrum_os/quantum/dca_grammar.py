"""DCA Grammar Gate (PLAN-23 §五) — validate that generated alternatives
respect the Crisis→Lag→Alternative→Direction recursion rules.

The DCA recursion grammar constrains how roles can transition:
  - crisis → lag | alternative | crisis (deepen)
  - lag → crisis | alternative | lag (persist)
  - alternative → direction | alternative (branch)
  - direction → crisis | lag | direction (persist)

Two transitions are forbidden:
  - direction → alternative (direction is a commitment, not revisable)
  - lag → direction (lag needs alternative mediation before becoming direction)
"""

DCA_GRAMMAR: dict = {
    "transitions": {
        "crisis": ["lag", "alternative", "crisis"],
        "lag": ["crisis", "alternative", "lag"],
        "alternative": ["direction", "alternative"],
        "direction": ["crisis", "lag", "direction"],
    },
    "forbidden": [
        ("direction", "alternative"),
        ("lag", "direction"),
    ],
    "recursion_depth": 5,
}

_VALID_ROLES = frozenset({"crisis", "lag", "alternative", "direction"})


def validate_alternative(current_role: str, proposed_role: str,
                         depth: int = 0) -> bool:
    """Check whether *proposed_role* is a valid transition from *current_role*.

    Args:
        current_role: The role we are transitioning from.
        proposed_role: The role we want to transition to.
        depth: Current recursion depth (for alternative-branch gating).

    Returns:
        ``True`` if the transition is allowed.
    """
    if current_role not in _VALID_ROLES or proposed_role not in _VALID_ROLES:
        return False

    # 1. Check forbidden list
    if (current_role, proposed_role) in DCA_GRAMMAR["forbidden"]:
        return False

    # 2. Check allowed transitions
    if proposed_role in DCA_GRAMMAR["transitions"].get(current_role, []):
        # 3. Recursion-depth gate: beyond max depth, only allow direction
        if depth > DCA_GRAMMAR["recursion_depth"]:
            return proposed_role == "direction"
        return True

    return False


def validate_alternative_chain(role_chain: list[str]) -> tuple[bool, str]:
    """Validate an entire chain of role transitions.

    Args:
        role_chain: Ordered list of role strings, e.g.
            ``["crisis", "lag", "alternative", "direction"]``.

    Returns:
        A ``(is_valid, first_failure_reason)`` tuple.  When the chain is
        valid the reason is an empty string.
    """
    if len(role_chain) < 2:
        # A single role or empty chain is trivially valid
        return True, ""

    for i in range(len(role_chain) - 1):
        current = role_chain[i]
        proposed = role_chain[i + 1]
        if not validate_alternative(current, proposed, depth=i):
            return False, (
                f"Invalid transition at position {i}: "
                f"'{current}' → '{proposed}'"
            )

    return True, ""


def mutate_alternative(current_role: str, depth: int = 0) -> list[str]:
    """Return the valid alternative roles that can follow *current_role*.

    Uses ``DCA_GRAMMAR.transitions`` and filters forbidden transitions as
    well as the recursion-depth gate.

    Args:
        current_role: The role to mutate from.
        depth: Current recursion depth (for alternative-branch gating).

    Returns:
        List of valid next-role strings.
    """
    if current_role not in _VALID_ROLES:
        return []

    candidates = DCA_GRAMMAR["transitions"].get(current_role, [])

    result: list[str] = []
    for candidate in candidates:
        if validate_alternative(current_role, candidate, depth=depth):
            result.append(candidate)

    return result
