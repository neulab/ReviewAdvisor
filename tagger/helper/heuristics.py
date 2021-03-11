# %%
from typing import List
import jsonlines
import fire


def pair_sent_with_id(id_file, sent_file):
    """ pair sentences with their corresponding id """
    paper_id_list = []
    with open(id_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            paper_id = line.strip()
            paper_id_list.append(paper_id)

    sent_list = []  # List[List[Tuple]]
    sent = []
    with open(sent_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                sent_list.append(sent)
                sent = []
            else:
                token, aspect = line.split(" ")
                sent.append((token, aspect))
    if len(sent) > 0:
        sent_list.append(sent)

    assert len(paper_id_list) == len(sent_list)

    # concate the id into the sent list
    sent_with_id = []
    for id, sent_elem in zip(paper_id_list, sent_list):
        list_to_add = [id] + sent_elem
        sent_with_id.append(list_to_add)

    return sent_with_id


def reconstruct_doc(sent_with_id: List):
    # concate every sentence for the same document
    doc_with_id = []
    current_id = sent_with_id[0][0]
    current_sent = sent_with_id[0]
    for l in sent_with_id[1:]:
        id = l[0]
        if id == current_id:
            current_sent += l[1:]
        else:
            doc_with_id.append(current_sent)
            current_id = id
            current_sent = l

    if len(current_sent) > 0:
        doc_with_id.append(current_sent)
    return doc_with_id


def combine(tag_list):
    # combine same consecutive tags
    new_tag_list = []
    current_tag = tag_list[0][0]
    start = tag_list[0][1]
    end = tag_list[0][2]
    for elem in tag_list:
        if elem[0] == current_tag:
            end = elem[2]
        else:
            new_tag_list.append([current_tag, start, end])
            current_tag = elem[0]
            start = elem[1]
            end = elem[2]
    new_tag_list.append([current_tag, start, end])
    return new_tag_list


def endsWithPunctuation(token: str) -> bool:
    if token.endswith('.') or token.endswith(','):
        return True
    else:
        return False


def is_special_symbol(token: str) -> bool:
    if len(token) == 1:
        if not token.isdigit():
            if not token.isalpha():
                if token != "'":
                    return True
    return False


def heuristics(doc_with_id_elem):
    """ take a List[id, Tuple], return with a same format """
    paper_id = doc_with_id_elem[0]
    words_with_labels = doc_with_id_elem[1:]
    words = [x[0] for x in words_with_labels]
    labels = [x[1] for x in words_with_labels]

    # convert to [tag, start, end] format
    current_label = labels[0]
    label_list = []
    start = 0
    end = 0

    for i, label in enumerate(labels):
        if label == current_label:
            end = i
        else:
            span = [current_label, start, end]
            label_list.append(span)
            current_label = label
            start = i
            end = i

    span = [current_label, start, end]
    label_list.append(span)

    # Heuristic 1: If there are no other tags (they are tagged as O which
    # stands for Outside) between two summary tags, then replace all tags
    # between them with summary tag.
    if len(label_list) >= 3:
        for i in range(len(label_list) - 2):
            if label_list[i][0] == 'summary' \
                    and label_list[i + 2][0] == 'summary' \
                    and label_list[i + 1][0] == 'O':
                label_list[i + 1][0] = 'summary'

    label_list = combine(label_list)

    # Heuristic 2: If there are multiple text spans tagged as summary,
    # keep the first one and discard others.
    summary_appear = False
    for i in range(len(label_list)):
        if label_list[i][0] == 'summary' and not summary_appear:
            summary_appear = True
            continue
        if summary_appear:
            if label_list[i][0] == 'summary':
                label_list[i][0] = 'O'

    label_list = combine(label_list)

    # Heuristic 3: If the punctuation is separately tagged and is
    # different from its neighbor, we replace its tag to O.
    for i in range(len(label_list)):
        if label_list[i][1] == label_list[i][2]:
            current_word = words[label_list[i][1]]
            if current_word == ',' or current_word == '.':
                label_list[i][0] = 'O'

    label_list = combine(label_list)

    # Heuristic 4 & 5
    if len(label_list) >= 3:
        # Heuristic 4: If two tags are separated by a single other tag,
        # then replace this tag with its right neighbor's tag.
        for i in range(len(label_list) - 2):
            if label_list[i][0] != 'summary' and label_list[i][0] != 'O' \
                    and label_list[i][0] == label_list[i + 2][0] \
                    and label_list[i + 1][0] != label_list[i][0]:
                if label_list[i + 1][0] != 'O' or label_list[i + 1][1] == label_list[i + 1][2]:
                    label_list[i + 1][0] = label_list[i + 2][0]
        label_list = combine(label_list)

        # Heuristic 5: If there exists a single token with a tag and its neighbors are O,
        # then replace this tag to O.
        for i in range(1, len(label_list) - 1):
            if label_list[i][0] != 'O' and label_list[i][1] == label_list[i][2] \
                    and label_list[i - 1][0] == 'O' and label_list[i + 1][0] == 'O':
                label_list[i][0] = 'O'
        label_list = combine(label_list)

    # Heuristic 6: For a non-summary non-O tag span, if its neighbors are O
    # and the start/end of this span is not special symbol (for example,
    # punctuations or other symbols that have 1 length), then we expand
    # from its start/end until we meet other non-O tag or special symbol.
    new_labels = []
    for elem in label_list:
        new_labels += [elem[0]] * (elem[2] - elem[1] + 1)

    if len(label_list) >= 3:
        for i in range(1, len(label_list) - 1):
            if label_list[i][0] != 'O' and label_list[i][0] != 'summary':
                start = label_list[i][1]
                end = label_list[i][2]
                # from start
                if label_list[i - 1][0] == 'O' and start > 0 \
                        and not is_special_symbol(words[start - 1]) \
                        and not endsWithPunctuation(words[start - 1]):
                    new_start = start
                    while new_start > 0 and not is_special_symbol(words[new_start - 1]) \
                            and not endsWithPunctuation(words[new_start - 1]) \
                            and new_labels[new_start - 1] == 'O':
                        new_start -= 1
                    label_list[i - 1][2] = new_start - 1
                    label_list[i][1] = new_start
                    # may exist end < start tag, we will delete them later
                # from end
                if label_list[i + 1][0] == 'O' and end < len(words) - 1 \
                        and not is_special_symbol(words[end]) \
                        and not endsWithPunctuation(words[end]):
                    new_end = end
                    while new_end < len(words) - 1 and not is_special_symbol(words[new_end + 1]) \
                            and not endsWithPunctuation(words[new_end]) \
                            and new_labels[new_end + 1] == 'O':
                        new_end += 1
                    label_list[i + 1][1] = new_end + 1
                    label_list[i][2] = new_end
                    # may exist end < start tag, we will delete them later
                # Since we didn't update the new_tags, we should solve conflicts later where the
                # end of the previous tag is greater than the start of current tag.

    new_label_list = []
    for elem in label_list:
        if elem[1] <= elem[2]:
            # Only keep those with start <= end
            new_label_list.append(elem)

    # Make corrections to the boundaries
    if len(new_label_list) >= 2:
        for i in range(len(new_label_list) - 1):
            new_label_list[i][2] = new_label_list[i + 1][1] - 1

    labels = []
    for elem in new_label_list:
        labels += [elem[0]] * (elem[2] - elem[1] + 1)

    # Heuristic 7: If the summary span does not end with a period, then we truncate or
    # extend it at most five words to make it ends with a period.
    summary_end = None
    for i in range(1, len(words) - 1):
        current_label = labels[i]
        next_label = labels[i + 1]
        if current_label == 'summary' and next_label != 'summary':
            summary_end = i

    if summary_end is not None:
        if words[summary_end].endswith('.') or words[summary_end + 1] == '.':
            # The boundary is correct
            pass
        else:
            i = summary_end
            new_summary_end = None
            left_count = 0
            right_count = 0
            while i >= 0 and not words[i].endswith('.'):
                # Left find first
                i -= 1
                left_count += 1
                if left_count == 6:
                    break
            if i > summary_end - 6 and i != -1:
                new_summary_end = i

            if new_summary_end is None:
                i = summary_end
                while i < len(words) - 1 and not words[i].endswith('.'):
                    # Then find right
                    i += 1
                    right_count += 1
                    if right_count == 6:
                        break
                if i < summary_end + 6:
                    new_summary_end = i
            if new_summary_end is not None and new_summary_end != summary_end:
                if new_summary_end < summary_end:
                    for idx in range(new_summary_end + 1, summary_end + 1):
                        labels[idx] = 'O'
                else:
                    for idx in range(summary_end, new_summary_end + 1):
                        labels[idx] = 'summary'

    return_list = [(x, y) for x, y in zip(words, labels)]
    return_list = [paper_id] + return_list

    return return_list


def postprocess(doc_with_id):
    new_doc_with_id = []
    for elem in doc_with_id:
        new_doc_with_id.append(heuristics(elem))
    return new_doc_with_id


def get_jsonlines(new_doc_with_id):
    lines = []
    for doc in new_doc_with_id:
        id = doc[0]
        text_list = []
        text_list += [t[0] for t in doc[1:]]
        text = " ".join(text_list)
        # text = detokenizer.detokenize(text_list)
        # we have text, then we need to find appropriate span
        labels = []
        current_label = doc[1][1]
        current_token = doc[1][0]
        pointer = 0
        label_start = pointer + text[pointer:].find(current_token)
        label_end = label_start + len(current_token)
        for token, label in doc[1:]:
            token_start = pointer + text[pointer:].find(token)
            pointer = token_start + len(token)  # represent the end of a token
            if label == current_label:
                label_end = pointer
            else:
                if label_start < label_end and current_label != "O":
                    labels.append([label_start, label_end, current_label])
                current_label = label
                label_start = pointer - len(token)
                label_end = pointer
        if label_start < label_end and current_label != "O":
            labels.append([label_start, label_end, current_label])
        # write a jsonl
        line = {"id": id, "text": text, "labels": labels}
        lines.append(line)
    return lines


def write_jsonlines(jsonl, jsonl_file):
    with jsonlines.open(jsonl_file, 'w') as writer:
        writer.write_all(jsonl)


def main(id_file, sent_file, jsonl_file):
    sent_with_id = pair_sent_with_id(id_file, sent_file)
    doc_with_id = reconstruct_doc(sent_with_id)
    new_doc_with_id = postprocess(doc_with_id)
    lines = get_jsonlines(new_doc_with_id)
    write_jsonlines(lines, jsonl_file)


if __name__ == '__main__':
    """ Convert sequence labeling format file into josnl file """
    # python heuristics.py test_aspect.id test_aspect.sent test_aspect.jsonl
    fire.Fire(main)
