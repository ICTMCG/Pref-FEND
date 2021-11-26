import enum
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import numpy as np
import json
import os
import torch
from config import INDEX2LABEL, INDEX_OF_LABEL, MAX_TOKENS_OF_A_POST
from utils import pytorch_cos_sim


def compute_classification_metrics(ans, pred, category_num=2):
    return classification_report(ans, pred, target_names=INDEX2LABEL, digits=4, output_dict=True)


def eval_when_training_on_single_gpu(outputs_file, dataset):
    # gt = np.array([x[-1].item() for x in tqdm(dataset)])
    gt = dataset.labels

    wrong_cls_cases = []
    classifying_ans = np.array([], dtype=int)
    classifying_pred = np.array([], dtype=int)

    out = json.load(open(outputs_file, 'r'))
    outputs = []
    for o in out:
        outputs += o

    for o in outputs:
        # o: [idx, 0_class_score, 1_class_score, 2_class_score]
        idx, scores = o[0], np.array(o[1:])
        pred = int(scores.argmax())
        ans = int(gt[idx])

        if ans != pred:
            case = {'idx': idx, 'label': INDEX2LABEL[ans],
                    'prediction': INDEX2LABEL[pred], 'prediction_scores': list(scores)}
            wrong_cls_cases.append(case)

        classifying_ans = np.append(classifying_ans, ans)
        classifying_pred = np.append(classifying_pred, pred)

    class_report = compute_classification_metrics(classifying_ans, classifying_pred,
                                                  category_num=len(outputs[0]) - 1)

    res_file = outputs_file.replace('_outputs_', '_res_')
    wrong_cls_file = res_file.replace('_res_', '_wrong_cls_')
    json.dump({'classification': class_report}, open(res_file, 'w'), indent=4)
    json.dump(wrong_cls_cases, open(wrong_cls_file, 'w'), indent=4)

    return class_report['macro avg']['f1-score']


def evaluate(args, loader, model, criterion, dataset_type):
    print('Eval time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    model.eval()

    eval_loss = 0.

    eval_normal_loss = 0.
    eval_reversed_loss = 0.
    eval_preference_loss = 0.

    outputs = []

    preferenced_pieces_dict = dict()

    with torch.no_grad():
        for idxs, graphs_entity, graphs_pattern, graphs_others, nums_nodes, labels in tqdm(loader):
            PrefFEND_out = model(idxs, loader.dataset, graphs_entity,
                                 graphs_pattern, graphs_others, nums_nodes)
            mlp_out, mlp_reversed_out, map_entity, map_pattern = PrefFEND_out[:4]

            # mlp_out: (batch_size, category_num)
            normal_loss = criterion(mlp_out, labels.long().to(args.device))
            reversed_loss = torch.tensor([0.], device=args.device)
            preference_loss = torch.tensor([0.], device=args.device)

            if mlp_reversed_out is not None:
                reversed_loss = criterion(
                    mlp_reversed_out, abs(labels - 1).long().to(args.device))

            if map_entity is not None:
                # map_entity/map_pattern: (batch_size, max_nodes)
                # sim_matrix: (batch_size, batch_size)
                sim_matrix = pytorch_cos_sim(map_entity, map_pattern)
                # fill NAN
                sim_matrix = torch.where(torch.isnan(sim_matrix), torch.tensor(
                    [0.], device=args.device), sim_matrix)
                preference_loss = torch.mean(torch.diag(sim_matrix))

            if args.pattern_based_model == 'EANN_Text':
                event_labels = [
                    loader.dataset.event_labels[idx.item()] for idx in idxs]
                event_labels = torch.tensor(
                    event_labels, dtype=torch.long, device=args.device)

                event_out, event_reversed_out = PrefFEND_out[-2:]

                normal_event_loss = criterion(event_out, event_labels)
                reversed_event_loss = torch.tensor(
                    [0.], device=args.device)
                if event_reversed_out is not None:
                    reversed_event_loss = criterion(
                        event_reversed_out, event_labels)

                # Following EANN, the weight of "event loss" is 1
                normal_loss = normal_loss + args.eann_weight_of_event_loss * normal_event_loss
                reversed_loss = reversed_loss + args.eann_weight_of_event_loss * reversed_event_loss

            normal_loss *= args.weight_of_normal_loss
            reversed_loss *= args.weight_of_reversed_loss
            preference_loss *= args.weight_of_preference_loss
            loss = normal_loss + reversed_loss + preference_loss

            eval_loss += loss.item()
            eval_normal_loss += normal_loss.item()
            eval_reversed_loss += reversed_loss.item()
            eval_preference_loss += preference_loss.item()

            # (bz, category_num)
            output = mlp_out
            score = [[idxs[i].item()] + x for i,
                     x in enumerate(output.cpu().numpy().tolist())]
            outputs.append(score)

            for i, idx in enumerate(idxs):
                idx = idx.item()
                piece = loader.dataset.graphs[idx]['graph'][:MAX_TOKENS_OF_A_POST]

                if map_entity is None:
                    continue

                preferenced_piece = []
                for j, (word_type, word, _) in enumerate(piece):
                    entity_weight = map_entity[i][j].item()
                    pattern_weight = map_pattern[i][j].item()
                    preferenced_piece.append(
                        {'word': word, 'type': word_type, 'weight_of_entity': entity_weight, 'weight_of_pattern': pattern_weight})

                preferenced_pieces_dict[idx] = preferenced_piece

    file = os.path.join(args.save, dataset_type + '_outputs_' + str(args.current_epoch) + '.json')
    with open(file, 'w') as f:
        json.dump(outputs, f)

    # Preferenced Pieces
    if args.use_preference_map:
        piece_file = file.replace('_outputs_', '_pieces_')
        with open(piece_file, 'w') as f:
            json.dump(preferenced_pieces_dict, f, indent=4, ensure_ascii=False)

    # Macro F1 score
    classification_metrics = eval_when_training_on_single_gpu(
        file, loader.dataset)

    eval_losses = [eval_loss, eval_normal_loss,
                   eval_reversed_loss, eval_preference_loss]
    eval_losses = [l/len(loader) for l in eval_losses]

    return eval_losses, classification_metrics


def eval_average_two_outputs(experimental_dataset, output_file_1, output_file_2, model_1, model_2):
    graph_file = os.path.join(
        DATASET_DIR, '{}/data/graph_{}.json'.format(experimental_dataset, 'test'))
    with open(graph_file, 'r') as f:
        graphs = json.load(f)
    print('[Dataset File]\t{}, sz = {}'.format(graph_file, len(graphs)))

    labels = [p['label'] for p in graphs]
    labels = torch.tensor([INDEX_OF_LABEL[l] for l in labels])

    gt = labels
    wrong_cls_cases = []
    classifying_ans = np.array([], dtype=int)
    classifying_pred = np.array([], dtype=int)

    def _get_outputs(file):
        out = json.load(open(file, 'r'))
        outputs = []
        for o in out:
            outputs += o
        return outputs

    outputs_1 = _get_outputs(output_file_1)
    outputs_2 = _get_outputs(output_file_2)

    for i, o1 in enumerate(outputs_1):
        # o: [idx, 0_class_score, 1_class_score, 2_class_score]
        o2 = outputs_2[i]
        assert o1[0] == o2[0]
        idx = o1[0]

        s1 = np.array(o1[1:], dtype=np.float)
        s2 = np.array(o2[1:], dtype=np.float)
        scores = (s1 + s2) / 2

        pred = int(scores.argmax())
        ans = int(gt[idx])

        if ans != pred:
            case = {'idx': idx, 'label': INDEX2LABEL[ans],
                    'prediction': pred, 'prediction_scores': list(scores)}
            wrong_cls_cases.append(case)

        classifying_ans = np.append(classifying_ans, ans)
        classifying_pred = np.append(classifying_pred, pred)

    class_report = compute_classification_metrics(classifying_ans, classifying_pred,
                                                  category_num=len(outputs_1[0]) - 1)

    dir = 'ckpts/LogitsAvg'
    if not os.path.exists(dir):
        os.mkdir(dir)

    res_file = 'ckpts/LogitsAvg/{}_{}_res.json'.format(model_1, model_2)
    wrong_cls_file = res_file.replace('_res', '_wrong_cls')
    json.dump({'model_1': model_1, 'model_2': model_2, 'model_1_file': output_file_1,
               'model_2_file': output_file_2, 'classification': class_report}, open(res_file, 'w'), indent=4)
    json.dump(wrong_cls_cases, open(wrong_cls_file, 'w'), indent=4)

    return class_report['macro avg']['f1-score']
