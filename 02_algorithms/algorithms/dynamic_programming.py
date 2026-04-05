"""
algorithms/dynamic_programming.py
===================================
Classic dynamic programming problems with bottom-up and memoisation solutions.

Problems covered:
    - Fibonacci
    - 0/1 Knapsack
    - Longest Common Subsequence (LCS)
    - Coin Change (minimum coins)
    - Longest Increasing Subsequence (LIS)
"""

from __future__ import annotations

from functools import lru_cache


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------

def fibonacci_memo(n: int) -> int:
    """
    Return the nth Fibonacci number using memoisation.

    Time: O(n), Space: O(n)
    """
    memo: dict[int, int] = {}

    def _fib(k: int) -> int:
        if k <= 1:
            return k
        if k not in memo:
            memo[k] = _fib(k - 1) + _fib(k - 2)
        return memo[k]

    return _fib(n)


def fibonacci_tabulation(n: int) -> int:
    """
    Return the nth Fibonacci number using bottom-up tabulation.

    Time: O(n), Space: O(1)
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ---------------------------------------------------------------------------
# 0/1 Knapsack
# ---------------------------------------------------------------------------

def knapsack_01(
    capacity: int,
    weights: list[int],
    values: list[int],
) -> tuple[int, list[int]]:
    """
    Solve the 0/1 Knapsack problem.

    Args:
        capacity: Maximum weight the knapsack can hold.
        weights:  Weight of each item.
        values:   Value of each item.

    Returns:
        (max_value, selected_indices) — the maximum total value achievable
        and the indices of chosen items.

    Time:  O(n * capacity)
    Space: O(n * capacity)
    """
    n = len(weights)
    # dp[i][w] = max value using first i items with capacity w
    dp: list[list[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i, v_i = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if w_i <= w:
                take = dp[i - 1][w - w_i] + v_i
                if take > dp[i][w]:
                    dp[i][w] = take

    # Back-track to find selected items
    selected: list[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected))


# ---------------------------------------------------------------------------
# Longest Common Subsequence
# ---------------------------------------------------------------------------

def lcs(s1: str, s2: str) -> str:
    """
    Return the Longest Common Subsequence of *s1* and *s2*.

    Time:  O(m * n)
    Space: O(m * n)
    """
    m, n = len(s1), len(s2)
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS string
    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))


def lcs_length(s1: str, s2: str) -> int:
    """Return only the length of the LCS (space-optimised: O(min(m,n)))."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# Coin Change (minimum coins)
# ---------------------------------------------------------------------------

def coin_change(coins: list[int], amount: int) -> int:
    """
    Return the fewest number of coins needed to make *amount*, or -1.

    Time:  O(amount * len(coins))
    Space: O(amount)
    """
    INF = float("inf")
    dp: list[float] = [INF] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)

    return int(dp[amount]) if dp[amount] != INF else -1


def coin_change_ways(coins: list[int], amount: int) -> int:
    """
    Return the number of distinct combinations that make *amount*.

    Time:  O(amount * len(coins))
    Space: O(amount)
    """
    dp: list[int] = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]


# ---------------------------------------------------------------------------
# Longest Increasing Subsequence
# ---------------------------------------------------------------------------

def lis_length(arr: list[int]) -> int:
    """
    Return the length of the Longest Increasing Subsequence.

    Uses patience sorting for O(n log n).
    """
    import bisect
    tails: list[int] = []
    for x in arr:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def lis(arr: list[int]) -> list[int]:
    """
    Return one Longest Increasing Subsequence.

    Time: O(n log n)
    """
    import bisect
    n = len(arr)
    if n == 0:
        return []

    tails: list[int] = []
    predecessor: list[int] = [-1] * n
    indices: list[int] = []

    for i, x in enumerate(arr):
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            indices.append(i)
        else:
            tails[pos] = x
            indices[pos] = i
        predecessor[i] = indices[pos - 1] if pos > 0 else -1

    # Backtrack from last index in indices
    result: list[int] = []
    k = indices[-1]
    while k != -1:
        result.append(arr[k])
        k = predecessor[k]
    return list(reversed(result))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate DP algorithms."""
    print("=== Fibonacci ===")
    for n in [0, 1, 5, 10, 30]:
        print(f"  fib({n}) = {fibonacci_tabulation(n)}")

    print("\n=== Knapsack 0/1 ===")
    weights = [2, 3, 4, 5]
    values  = [3, 4, 5, 6]
    capacity = 8
    max_val, chosen = knapsack_01(capacity, weights, values)
    print(f"  Max value: {max_val}, items: {chosen}")

    print("\n=== LCS ===")
    pairs = [("ABCBDAB", "BDCAB"), ("AGGTAB", "GXTXAYB")]
    for a, b in pairs:
        print(f"  LCS('{a}', '{b}') = '{lcs(a, b)}'")

    print("\n=== Coin Change ===")
    print(f"  Min coins for 11 with [1,5,6,9]: {coin_change([1, 5, 6, 9], 11)}")
    print(f"  Ways to make 5 with [1,2,5]:     {coin_change_ways([1, 2, 5], 5)}")

    print("\n=== LIS ===")
    seq = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"  LIS of {seq}")
    print(f"  Length: {lis_length(seq)}, sequence: {lis(seq)}")


if __name__ == "__main__":
    main()
