def count_token(query, iterable):
    """
    >>> count_tokens(1, [1, 2, 3, 1])
    2
    """
    return sum(1 for word in iterable if word == query)


def count_prefix(query, iterable):
    return sum(1 for word in iterable if word is not None and word.startswith(query))
