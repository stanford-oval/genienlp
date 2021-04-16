import ujson
with open(f'99/chunked/merged/bootleg/bootleg_uncased_mini/prepped_files/eval_bootleg/bootleg_wiki/bootleg_labels.jsonl', 'r') as fin:
    all_lines = fin.readlines()
    all_sent_ids = [ujson.loads(line)['sent_idx_unq'] for line in all_lines]
    all_lines = list(zip(*sorted(zip(all_sent_ids, all_lines), key=lambda item: item[0])))[1]
    i = 0
    for line in all_lines:
        if i < 10:
            print(line)
        i += 1
print()
