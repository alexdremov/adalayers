def tokenize_and_align_labels(tokenizer, words, tags):
    tokenized_inputs = tokenizer(words, truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()

    current_word = None
    label_ids = []
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word/entity
            current_word = word_id

            #  Assign -100 to labels for special tokens, else use the word's label
            label = -100 if word_id is None else tags[word_id]

            # Append the adjusted label to the new_labels list
            label_ids.append(label)
        elif word_id is None:
            # Handle special tokens by assigning them a label of -100
            label_ids.append(-100)
        else:
            # Token belongs to the same word/entity as the previous token
            label = tags[word_id]

            # If the label is in the form B-XXX, change it to I-XXX
            if label % 2 == 1:
                label += 1

            # Append the adjusted label to the new_labels list
            label_ids.append(label)

    tokenized_inputs["word_ids"] = list(word_ids)
    tokenized_inputs["labels"] = list(label_ids)
    return tokenized_inputs


def collapse_tokenized_token_predictions(word_ids, predictions):
    result = []
    for word_id, prediction in zip(word_ids, predictions):
        if word_id is None:
            continue
        if len(result) > word_id:
            continue
        result.append(prediction)
    return result
