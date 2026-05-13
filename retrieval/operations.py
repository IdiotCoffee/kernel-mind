from retrieval.tokenize import tokenize

# =========================================================
# Operation Vocabulary
# =========================================================

CRUD_OPERATIONS = {
    "create",
    "read",
    "update",
    "delete",
    "get",
    "list",
}

AUTH_OPERATIONS = {
    "login",
    "logout",
    "authenticate",
    "verify",
    "token",
    "reset",
    "password",
    "register",
}

ALL_OPERATIONS = CRUD_OPERATIONS | AUTH_OPERATIONS


# =========================================================
# Extract Operations From Query
# =========================================================


def extract_operations(query: str):

    tokens = set(tokenize(query))

    return tokens.intersection(ALL_OPERATIONS)


# =========================================================
# Operation Match Score
# =========================================================


def compute_operation_match_score(
    query: str,
    fqn: str,
) -> float:
    """
    Strong symbolic anchoring for:
    - CRUD intent
    - auth intent
    - workflow specificity

    Helps prevent:
    - semantic sibling overspill
    - CRUD flooding
    """

    query_ops = extract_operations(query)

    if not query_ops:
        return 0.0

    fqn_tokens = set(tokenize(fqn))

    overlap = query_ops.intersection(fqn_tokens)

    if not overlap:
        return -0.35

    return len(overlap) * 1.25
