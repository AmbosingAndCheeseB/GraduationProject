import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import  BertForMaskedLM
import re



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

parser = argparse.ArgumentParser()

parser.add_argument('--problem', type = str, required=True)

args = parser.parse_args()
p_list = args.problem.split('/')

def get_score(model, tokenizer, input_tensor, segment_tensor, masked_index, candidate):
    candidate_tokens = tokenizer.tokenize(candidate)
    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)
    predictions = model(input_tensor, segment_tensor)
    predictions_candidates = predictions[0, masked_index, candidate_ids].mean()

    del input_tensor
    del segment_tensor

    return predictions_candidates.item()


try:
    problem = re.sub('\_+', ' [MASK] ', p_list[0])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_texts = tokenizer.tokenize(problem)

    masked_idx = tokenized_texts.index("[MASK]")

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_texts)

    segment_ids = [0] * len(tokenized_texts)

    problem_tensors = torch.tensor([input_ids]).to(device)
    segment_tensors = torch.tensor([segment_ids]).to(device)
    candidates = [p_list[1], p_list[2], p_list[3], p_list[4]]

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.cuda()

    PATH = "./epoch10.pt"

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logit = torch.tensor([get_score(model, tokenizer, problem_tensors , segment_tensors, masked_idx, candi) for candi in candidates])
        logit_idx = torch.argmax(logit).item()
        if(logit_idx == 0):
            print("(A)" + candidates[logit_idx])
        elif(logit_idx == 1):
            print("(B)" + candidates[logit_idx])
        elif(logit_idx == 2):
            print("(C)" + candidates[logit_idx])
        elif(logit_idx == 3):
            print("(D)" + candidates[logit_idx])

except Exception as ex:
    print(ex)