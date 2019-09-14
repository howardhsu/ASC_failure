
import os
import json
import glob
import random
import numpy as np
import torch
from collections import namedtuple
import asclab
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


class RunManager:    
    def batch_run(self, config_dir):
        configs = glob.glob(os.path.join(config_dir, "*.json") )
        for cfg in configs:
            with open(cfg) as f:
                config = json.load(f)
            self.single_run(config)
    
    def succ(self, run_config):
        for test_file in run_config.test_file:
            if not os.path.exists( os.path.join(run_config.output_dir, "predictions_" + test_file)):
                return False
        return True
    
    def single_run(self, config):
        for seed in range(1, config["run"] + 1):
            run_config = dict(config)
            run_config["seed"] = seed
            run_config["output_dir"] = os.path.join(run_config["output_dir"], str(seed))

            run_config = namedtuple("run_config", run_config.keys())(*run_config.values())

            os.makedirs(run_config.output_dir, exist_ok = True)
            if self.succ(run_config):
                continue
            
            with open(os.path.join(run_config.output_dir, "config.json"), "w") as fw:
                json.dump(config, fw)
            
            random.seed(run_config.seed)
            np.random.seed(run_config.seed)
            torch.manual_seed(run_config.seed)
            
            trainer = getattr(asclab, run_config.trainer)()
            trainer.train(run_config)

            for test_file in run_config.test_file:
                evalmode = test_file.split("|")[0]
                test_file = test_file.split("|")[1]
                predictor = asclab.Predictor()
                predictor.test(test_file, evalmode, run_config)
            if run_config.remove_model:
                os.remove(os.path.join(run_config.output_dir, "model.pt"))
                

if __name__=="__main__":
    mgr = RunManager()
    mgr.batch_run("configs")