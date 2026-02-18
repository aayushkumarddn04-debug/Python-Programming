def is_arithmetic_progression(sequence):
    """
    Checks if the given sequence is an arithmetic progression.
    An arithmetic progression has a constant difference between consecutive terms.
    """
    if len(sequence) < 2:
        return True  # A sequence with less than 2 elements is trivially an AP

    # Calculate the common difference
    common_diff = sequence[1] - sequence[0]

    # Check if the difference is consistent for all consecutive pairs
    for i in range(2, len(sequence)):
        if sequence[i] - sequence[i-1] != common_diff:
            return False

    return True

# Example usage with sequences from lab2.ipynb
sequences = [
    [1, 2, 3, 4, 5],  # Should be True
    [2, 5, 7, 9, 12, 16],  # Should be False (differences: 3,2,2,3,4 - not constant)
    [20, 30, 12, 12, 14, 14],  # Should be False
    list(range(0, 5)),  # [0,1,2,3,4] - True
    list(range(-100, 100, 2))  # Even numbers from -100 to 98 - True
]

for seq in sequences:
    print(f"Sequence: {seq}")
    print(f"Is arithmetic progression: {is_arithmetic_progression(seq)}")
    print()
