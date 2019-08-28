import argparse
import json
import numpy as np
import os

def build_weights(args):
    incorrect=[]
    ids=[]
    for run in range(1, args.run+1):
        with open(os.path.join(args.run_dir, args.domain, str(run), args.pred_file) ) as f:
            preds=json.load(f)
            y_pred=np.array(preds['logits']).argmax(axis=-1)   
            y_true=np.array(preds['label_ids'])
            incorrect.append(y_pred!=y_true)
            if len(ids)>0:
                assert preds['ids']==ids
            ids=preds['ids']

    weights=1.+ np.log(sum(incorrect)+1. ) #try exp later.

    with open(args.src_json) as f:
        data=json.load(f)

    for ix, id in enumerate(ids):
        data[id]['weight']=float(weights[ix] )

    with open(args.tgt_json, "w") as fw:
        json.dump(data, fw, sort_keys=True, indent=4)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None, type=str, required=True)
    parser.add_argument("--domain", default='laptop', type=str)
    parser.add_argument("--run", default=10, type=int)
    parser.add_argument("--pred_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input json files.")

    parser.add_argument("--src_json",
                        default=None,
                        type=str,
                        required=True,
                        help="The original testing data.")
    
    parser.add_argument("--tgt_json",
                        default=None,
                        type=str,
                        required=True,
                        help="The original testing data.")
    
    args = parser.parse_args()
    build_weights(args)

if __name__=="__main__":
    main()