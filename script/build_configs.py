
import json

global_config = {
    "max_seq_length": 100,
    "train_batch_size": 32,
    "learning_rate": 3e-5, 
    "run": 10,
    "eval_batch_size": 8,
    "warmup_proportion": 0.1,
    "remove_model": True,
}

domain_config = {
    "laptop": {
        "bert_model": "laptop_pt_review",
        "data_dir": "asc/laptop",
    },
    "rest": {
        "bert_model": "rest_pt_review",
        "data_dir": "asc/rest",
    }
}

baseline_config = {
    "pt_review": {
        "trainer": "Trainer",
        "model": "BaseClassifier",
        "num_train_epochs": 4,
        "test_file": ["asp|test_full.json", "asp|test_contra.json", "noasp|test_full.json"],
    },
    
    "manual_bal": {
        "trainer": "Manual_bal",
        "model": "ClassifierWithSampleWeight",
        "num_train_epochs": 4,
        "test_file": ["asp|test_full.json", "asp|test_contra.json"],
    },
    
    "focalloss": {
        "trainer": "Focalloss",
        "model": "ClassifierWithFocalLoss",
        "num_train_epochs": 4,
        "gamma": 2.0,
        "test_file": ["asp|test_full.json", "asp|test_contra.json"],
    },
    
    "adaweight_-0.05": {
        "trainer": "AdaWeight",
        "model": "ClassifierWithSampleWeight",
        "num_train_epochs": 10,
        "factor": -0.05,
        "test_file": ["asp|test_full.json", "asp|test_contra.json"],
    },
    
    "adacontraweight": {
        "trainer": "AdaContraWeight",
        "model": "ClassifierWithSampleWeight",
        "num_train_epochs": 10,
        "factor": 0.0,
        "test_file": ["asp|test_full.json", "asp|test_contra.json"],
    },
}


for baseline in baseline_config:
    for domain in domain_config:
        config = dict(global_config)
        
        config.update(baseline_config[baseline])
        config.update(domain_config[domain])
        
        config_name = baseline + "_" + domain
        config["output_dir"] = "run/" + baseline + "/" + domain
        
        with open("configs/" + config_name + ".json", "w") as fw:
            json.dump(config, fw)
