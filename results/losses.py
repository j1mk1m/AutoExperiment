import json

def percent_loss(gold, pred):
    return abs(gold - pred) / gold

def collect_loss(gold, pred, metric_fn=percent_loss):
    loss = []
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except Exception as e:
            try:
                pred = int(pred)
            except Exception as e:
                print(f"Error in converting prediction: {pred}")
                return [1]

    if isinstance(pred, dict):
        for key in gold:
            if key not in pred:
                pred[key] = 0.0
            loss += collect_loss(gold[key], pred[key], metric_fn)
    elif isinstance(pred, list):
        for i in range(len(pred)):
            loss += collect_loss(gold[i], pred[i], metric_fn)
    else:
        try:
            loss += [metric_fn(gold, pred)]
        except Exception as e:
            print(f"Got error during loss calculation: {e}")
            return [1]
    return loss


def calculate_loss(gold, pred, metric_fn=percent_loss):
    loss_per_exp = {}
    correct_per_exp = {}
    correct_count = 0
    all_correct = True
    
    for key in gold:
        if key in pred:
            losses = collect_loss(gold[key], pred[key], metric_fn)
            loss = sum(losses) / len(losses)
        else:
            loss = 1
        loss_per_exp[key] = loss
        correct = loss <= 0.05
        if correct:
            correct_count += 1 
        correct_per_exp[key] = correct
        all_correct = all_correct and correct
    return loss_per_exp, correct_per_exp, correct_count, all_correct



