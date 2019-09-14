
import json
import os
import sklearn.metrics
import numpy as np
import glob
from collections import Counter, namedtuple


class Evaluator:
    def batch_eval(self, config_dir):
        configs = glob.glob(os.path.join(config_dir, "*.json") )
        scores = {}
        for cfg in configs:
            with open(cfg) as f:
                config = json.load(f)
            scores[cfg.split("/")[-1]] = self.single_run(cfg, config)
        with open("result.json", "w") as fw:
            json.dump(scores, fw, indent = 4)
    
    def succ(self, run_config):
        for test_file in run_config.test_file:
            if not os.path.exists( os.path.join(run_config.output_dir, "predictions_" + test_file)):
                return False
        return True
    
    def single_run(self, name, config):
        """evaluate on a single .json config file.
        """

        summary = {}
        for test_file in config["test_file"]:
            scores, cms = [], []
            for seed in range(1, config["run"]+1):
                run_config = dict(config)
                run_config["seed"] = seed
                run_config["output_dir"] = os.path.join(run_config["output_dir"], str(seed))
                run_config = namedtuple("run_config", run_config.keys())(*run_config.values())

                if not self.succ(run_config):
                    print("incomplete running of", name)
                    return

                fn = os.path.join(run_config.output_dir, "predictions_" + test_file)
                with open(fn) as f:
                    results=json.load(f)
                y_true=results['label_ids']
                y_pred=[np.argmax(logit) for logit in results['logits'] ]
                
                ####### accuracy ########
                acc = 100 * sklearn.metrics.accuracy_score(y_true, y_pred)
                
                ####### macro f1 ########
                p_macro, r_macro, f_macro, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
                f_macro = 100 * 2 * p_macro*r_macro/(p_macro+r_macro)
                
                ####### f1 per class ########
                _, _, f_label, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
                
                cm = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
                scores.append([acc, f_macro] + f_label.tolist())
                cms.append(cm)

            scores = np.array(scores).mean(axis=0)
            cms = np.vstack(cms).mean(axis=0)
            metric_name = {0: "acc", 1: "mf1", 2: "pos_f1", 3: "neg_f1", 4: "neu_f1"}
            scores_str = ", ".join([metric_name[ix] + "=" + str(round(score, 2)) for ix, score in enumerate(scores)])
            print(name, test_file, ":", scores_str)
            mtr = {metric_name[ix]: round(score, 2) for ix, score in enumerate(scores)}
            mtr["cm"] = cms.tolist()
            summary[test_file] = mtr
        return summary

            
if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.batch_eval("configs")