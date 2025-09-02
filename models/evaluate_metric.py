# -*- coding: utf-8 -*

def get_chunks_onesent(tag_sequence, sentid):
    """
    get the chunks from one sentence
    """
    chunks = []
    chunk_start, chunk_end = -1, -1
    chunk_type = None
    for i, tag in enumerate(tag_sequence):
        if tag.startswith('B-'):
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, chunk_end, sentid))
            chunk_start, chunk_end = i, i
            chunk_type = tag[2:]
        elif tag.startswith('I-') and tag[2:] == chunk_type:
            chunk_end = i
        elif tag == 'O':
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, chunk_end, sentid))
                chunk_type = None
        else:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, chunk_end, sentid))
            chunk_type = None
    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, chunk_end, sentid))
    return chunks


def evaluate_chunk_level(pred_chunks, true_chunks):
    """
    evaluate the chunk level
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    correct_preds += len(set(true_chunks) & set(pred_chunks))
    total_preds += len(pred_chunks)
    total_correct += len(true_chunks)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return f1, p, r, correct_preds, total_preds, total_correct


def evaluate_ByCategory(pred_chunks, true_chunks, classes):
    """
    evaluate the chunk level by category
    """
    class2f1 = {}
    for class_name in classes:
        pred_chunks_class = [chunk for chunk in pred_chunks if chunk[0] == class_name]
        true_chunks_class = [chunk for chunk in true_chunks if chunk[0] == class_name]
        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(pred_chunks_class, true_chunks_class)
        class2f1[class_name] = f1
    return class2f1