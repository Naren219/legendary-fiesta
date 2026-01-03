from symbolic_utilities import lt_eval_dict
from tqdm import tqdm
import torch
from torch import tensor
import numpy as np

def generate_answers(model, tokenizer, prompts, examples=None, eval_dict=None,
                     batch_size=8, max_new_tokens=128,
                     num_return_sequences=1, do_sample=False, temperature=0.7,
                     return_all=False):

    # Tokenize all prompts individually
    tokenized = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        tokenized.append((i, ids))

    # Sort by length (longest first) to minimize padding
    tokenized.sort(key=lambda x: len(x[1]), reverse=True)

    answers = [None] * len(tokenized)

    # Build generate kwargs
    generate_kwargs = dict(
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs['temperature'] = temperature

    for i in tqdm(range(0, len(tokenized), batch_size)):
        batch_items = tokenized[i:i+batch_size]
        indices = [x[0] for x in batch_items]
        seqs = [x[1] for x in batch_items]

        # Pad manually (left padding)
        max_len = max(len(s) for s in seqs)
        padded = []
        attention_masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append([tokenizer.pad_token_id] * pad_len + s)
            attention_masks.append([0] * pad_len + [1] * len(s))

        input_ids = tensor(padded, dtype=torch.int64).to('cuda')
        attention_mask = tensor(attention_masks, dtype=torch.int64).to('cuda')

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # outputs shape: (batch_size * num_return_sequences, seq_len)
        for j, idx in enumerate(indices):
            start = j * num_return_sequences
            end = start + num_return_sequences
            prompt_outputs = outputs[start:end]

            decoded = [
                tokenizer.decode(out[max_len:], skip_special_tokens=True)
                for out in prompt_outputs
            ]

            if num_return_sequences == 1:
                answers[idx] = decoded[0]
            elif return_all:
                answers[idx] = decoded
            else:
                # Select best candidate
                answers[idx] = select_best(decoded, examples[idx], eval_dict)

    return answers


def select_best(candidates, examples, eval_dict):
    """Select the candidate with highest accuracy on the examples."""
    best_answer = candidates[0]  # fallback
    best_score = -1

    for candidate in candidates:
        try:
            func = eval(candidate, eval_dict)
            score = sum(1 for inp, out in examples if func(inp) == out) / len(examples)
            if score > best_score:
                best_score = score
                best_answer = candidate
            if score == 1.0:
                break  # perfect match
        except:
            continue

    return best_answer

def get_accuracy(answers, test_data):
    """Compute accuracy on a given test dataset."""
    def safe_eval(answer, eval_dict):
        try:
            return eval(answer, eval_dict)
        except (SyntaxError, TypeError, NameError):
            return None

    functions = [safe_eval(a, lt_eval_dict) for a in answers]

    invalid_count = sum(1 for f in functions if f is None)
    print(f"Invalid syntax: {invalid_count}/{len(functions)} ({100*invalid_count/len(functions):.1f}%)")

    def check_accuracy(x, indices, functions):
        function = functions[indices]
        if function is None:
            return {'accuracy': 0.0}
        accuracy = 0
        for inp, out in x['examples']:
            try:
                if out == function(inp):
                    accuracy += 1/len(x['examples'])
            except Exception:
                pass
        return {'accuracy': accuracy}

    test_data = test_data.map(check_accuracy, fn_kwargs={'functions': functions}, with_indices=True)
    return np.mean(test_data['accuracy'])