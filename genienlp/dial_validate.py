import copy
import logging
import sys
import time
from collections import defaultdict

import torch
import ujson
from BiToD.evaluate import r_en_API_MAP
from BiToD.knowledgebase import api
from BiToD.knowledgebase.en_zh_mappings import API_MAP
from BiToD.utils import knowledge2span, read_require_slots, span2state, state2constraints, state2span

from genienlp.data_utils.example import NumericalizedExamples, SequentialField
from genienlp.util import GenerationOutput

logger = logging.getLogger(__name__)


def generate_with_seq2seq_model_for_dialogue_interactive(
    model,
    numericalizer,
    task,
    args,
    output_predictions_only=False,
    original_order=None,
    disable_progbar=True,
) -> GenerationOutput:
    from termcolor import colored

    bitod_preds = dict()

    predictions = []
    example_ids = []
    answers = []
    contexts = []

    device = model.device

    required_slots = read_require_slots()
    required_slots = {API_MAP[k]: v for k, v in required_slots.items()}
    api_names = list(required_slots.keys())

    dial_id = 'none'
    turn_id = 1
    dialogue_state = {}
    new_state_text = 'null'
    new_knowledge_text = 'null'
    active_api = None
    bitod_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}

    convo_history = []
    convo_window = 3

    hyperparameter_idx = 0

    train_target = 'response'

    next_target = {'dst': 'api', 'api': 'response', 'response': 'dst'}

    while True:
        try:
            batch_prediction = []

            # becomes dst for first turn
            train_target = next_target[train_target]

            if train_target == 'dst':
                if convo_history:
                    print(colored(f'SYSTEM: {convo_history[-1]}', 'red', attrs=['bold']))
                else:
                    print(colored('SYSTEM: Hello! What are you looking for today?', 'red', attrs=['bold']))

                # Hello, I am looking for a restaurant with Vegan Options.
                # construct new input
                raw_user_input = input(colored('USER: ', 'green', attrs=['bold']))
                if raw_user_input == 'RESET':
                    generate_with_seq2seq_model_for_dialogue_interactive(
                        model,
                        numericalizer,
                        task,
                        args,
                        output_predictions_only=False,
                        original_order=None,
                        disable_progbar=True,
                    )
                    break
                elif raw_user_input == 'END':
                    sys.exit(0)
                elif raw_user_input == 'STATE':
                    print(f'dialogue state: {dialogue_state}')
                    continue

                raw_user_input = 'USER: ' + raw_user_input.strip()

                convo_history.append(raw_user_input)

                input_text = f'DST: <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            elif train_target == 'api':
                new_state_text = state2span(dialogue_state, required_slots)

                # replace state
                input_text = f'API: <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            elif train_target == 'response':

                input_text = f'Response: <knowledge> {new_knowledge_text} <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            else:
                raise ValueError(f'Invalid train_target: {train_target}')

            tokenized_contexts = numericalizer.encode_batch([input_text], field_name='context', features=None)[0]

            numericalized_turn = NumericalizedExamples(
                example_id=[str(turn_id)],
                context=SequentialField(
                    value=torch.tensor([tokenized_contexts.value], device=device),
                    length=torch.tensor([tokenized_contexts.length], device=device),
                    limited=torch.tensor([tokenized_contexts.limited], device=device),
                    feature=None,
                ),
                answer=SequentialField(value=None, length=None, limited=None, feature=None),
            )

            generated = model.generate(
                numericalized_turn,
                max_output_length=args.max_output_length,
                num_outputs=args.num_outputs[hyperparameter_idx],
                temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0,
                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                top_k=args.top_k[hyperparameter_idx],
                top_p=args.top_p[hyperparameter_idx],
                num_beams=args.num_beams[hyperparameter_idx],
                num_beam_groups=args.num_beam_groups[hyperparameter_idx],
                diversity_penalty=args.diversity_penalty[hyperparameter_idx],
                no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                do_sample=args.temperature[hyperparameter_idx] != 0,
            )

            partial_batch_prediction_ids = generated.sequences

            partial_batch_prediction = numericalizer.reverse(partial_batch_prediction_ids, 'answer')[0]

            # post-process predictions
            partial_batch_prediction = task.postprocess_prediction(turn_id, partial_batch_prediction)

            # put them into the right array
            batch_prediction.append([partial_batch_prediction])

            predictions += batch_prediction

            if train_target == 'dst':
                # update dialogue_state
                lev = predictions[-1][0].strip()
                state_update = span2state(lev, api_names)
                for api_name in state_update:
                    active_api = api_name
                    if api_name not in dialogue_state:
                        dialogue_state[api_name] = state_update[api_name]
                    else:
                        dialogue_state[api_name].update(state_update[api_name])

                #### save latest state
                state_to_record = copy.deepcopy(dialogue_state)
                state_to_record = {r_en_API_MAP.get(k, k): v for k, v in state_to_record.items()}
                bitod_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
                ####

            elif train_target == 'api':
                new_knowledge_text = 'null'
                do_api_call = predictions[-1][0].strip()

                if do_api_call == 'yes':
                    # make api call
                    api_name = active_api

                    if api_name in dialogue_state:
                        constraints = state2constraints(dialogue_state[api_name])
                        # domain = api_name.split(" ")[0]
                        knowledge = defaultdict(dict)

                        try:
                            msg = api.call_api(r_en_API_MAP.get(api_name, api_name), constraints=[constraints])
                        except Exception as e:
                            logger.error(f'Error: {e}')
                            logger.error(
                                f'Failed API call with api_name: {api_name}, constraints: {constraints},'
                                f' processed_query: {msg[2]}, for turn: {dial_id}/{turn_id}'
                            )
                            msg = [0, 0, 0]

                        if int(msg[1]) <= 0:
                            logger.warning(
                                f'Message = No item available for api_name: {api_name}, constraints: {constraints},'
                                f' processed_query: {msg[2]}, for turn: {dial_id}/{turn_id}'
                            )

                            new_knowledge_text = f'( {api_name} ) Message = No item available.'
                        else:
                            # always choose highest ranking results (having deterministic api results)
                            knowledge[api_name].update(msg[0])
                            new_knowledge_text = knowledge2span(knowledge)

                        #### save latest api constraints
                        bitod_preds[dial_id]["API"][r_en_API_MAP.get(api_name, api_name)] = copy.deepcopy(constraints)
                        ####

                #### save latest api results and constraints
                bitod_preds[dial_id]["turns"][str(turn_id)]["api"] = new_knowledge_text
                ####

            if train_target == 'response':
                #### save latest response
                bitod_preds[dial_id]["turns"][str(turn_id)]["response"] = predictions[-1]
                ####

                convo_history.append('SYSTEM: ' + predictions[-1][0])

        except KeyboardInterrupt:
            break

    with open(f"{int(time.time())}_bitod_preds.json", 'w') as fout:
        ujson.dump(bitod_preds, fout, indent=2, ensure_ascii=False)

    if original_order is not None:
        # sort back to the original order
        original_order, example_ids, predictions, answers, contexts = [
            list(a) for a in tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts)))))
        ]

    # TODO calculate and return loss
    loss = None
    output = GenerationOutput(loss=loss)

    if output_predictions_only:
        output.predictions = predictions
    else:
        output.example_ids, output.predictions, output.answers, output.contexts = example_ids, predictions, answers, contexts

    return output
