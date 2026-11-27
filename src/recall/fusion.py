def fuse(channels, offset, k):
    total = {}
    for ranked in channels:
        for rank, (item, _) in enumerate(ranked, 1):
            total[item] = total.get(item, 0.0) + 1.0 / (offset + rank)
    return sorted(total, key=lambda item: (-total[item], item))[:k]
