from neurograph.models.document import Chunk


def select_context(
    results: list[tuple[Chunk, float]],
    max_distance: float,
    max_chars: int = 4000,
) -> list[Chunk]:
    selected: list[Chunk] = []
    total_chars = 0

    for chunk, distance in results:
        if distance > max_distance:
            continue
        chunk_size = len(chunk.text)
        if total_chars + chunk_size > max_chars:
            continue
        selected.append(chunk)
        total_chars += chunk_size

    return selected
