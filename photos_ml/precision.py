def precision_at_k(output, desired_output, k):
    valid = [False for _ in range(k)]
    for i, out in enumerate(output[:k]):
        if out in desired_output:
            valid[i] = True
    return sum(valid) / min(len(output), k)


def mean_average_precision(output, desired_output):
    ap = 0
    valid_no = 0
    for i, out in enumerate(output):
        if out in desired_output:
            valid_no += 1
            ap += valid_no / (i + 1)

    return ap / valid_no if valid_no else 0
