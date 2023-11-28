import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    print(f"layers: {layers}")
    print(f"consts: {consts}")
    n = len(layers)
    res = indeps
    print(f"res: {res}")
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        print(f"Intermediate result at layer {i}: {res}")  # Debugging print statement
        if i!=n-1: res = F.relu(res)
    print(f"Final result: {res}")  # Debugging print statement
    return torch.sigmoid(res)