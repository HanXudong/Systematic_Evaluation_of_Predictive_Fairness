import itertools as it
import numpy as np
import secrets

DIST_TYPE = "g_cond_y"
# DIST_TYPE = "joint"

def loguniform(low=0, high=1, size=None):
    return np.power(10, np.random.uniform(low, high, size))

def log_grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.power(10, np.array([start+i*step for i in step_index]))

def grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.array([start+i*step for i in step_index])

yaml_opt = {
    "code_dir": "/DIR_TO_FAIRLIB",
    "code_api": "fairlib",
    "results_dir": "experimental_results/",
}



slurm_head = """#!/bin/bash

cd {code_dir}

"""

def write_to_batch_files(job_name, exps, allNames, file_path="scripts/dev/", additional_args_list=[""]):

    for _dataset in exps["dataset"]:
        with open(file_path+"{}_{}.slurm".format(_dataset, job_name),"w") as f:
            f.write(slurm_head.format(
                _job_name = _dataset+"_"+job_name,
                code_dir = yaml_opt["code_dir"]
                ))

    for a_id, additional_args in enumerate(additional_args_list):
        combos = it.product(*(exps[Name] for Name in allNames))
        for id, combo in enumerate(combos): 
            _dataset = combo[0]       
            with open(file_path+"{}_{}.slurm".format(_dataset, job_name),"a+") as f:
                command = "python {code_api} --project_dir {_project_dir} --dataset {_dataset} --emb_size {_emb_size} --num_classes {_num_classes} --batch_size {_batch_size} --lr {_learning_rate} --hidden_size {_hidden_size} --n_hidden {_n_hidden} --dropout {_dropout}{_batch_norm} --base_seed {_random_seed} --exp_id {_exp_id} --epochs_since_improvement 10 --num_groups {_num_groups} --epochs 50 --results_dir {results_dir} --GBT --GBTObj {DIST_TYPE} --GBT_N 30000 --GBT_alpha {_GBT_alpha} {_additional_args}"
                # dataset
                _dataset = combo[0]
                _emb_size = 2304 if _dataset == "Moji" else 768
                _num_classes = 2 if _dataset == "Moji" else 28
                _num_groups = 4 if _dataset == "Bios_both" else 2
                _batch_size = combo[1]
                _learning_rate = combo[2]
                _hidden_size = combo[3]
                _n_hidden = combo[4]
                _dropout = combo[5]
                _batch_norm = " --batch_norm" if combo[6] else ""
                _random_seed = combo[7]
                _project_dir = combo[8]
                _GBT_alpha = combo[9]

                _exp_id = "{_job_name}_{_GBT_alpha}_{_random_seed}_{_additional_id}".format(
                    _GBT_alpha = _GBT_alpha,
                    _job_name = job_name,
                    _random_seed=_random_seed,
                    _additional_id = a_id,
                    )

                command=command.format(
                    code_api = yaml_opt["code_api"],
                    _dataset=_dataset,
                    _emb_size=_emb_size,
                    _num_classes=_num_classes,
                    _batch_size=_batch_size,
                    _learning_rate=_learning_rate,
                    _hidden_size=_hidden_size,
                    _n_hidden=_n_hidden,
                    _dropout=_dropout,
                    _batch_norm=_batch_norm,
                    _random_seed=(_random_seed+secrets.randbelow(int(1e7))),
                    _exp_id=_exp_id,
                    _project_dir=_project_dir,
                    _num_groups = _num_groups,
                    results_dir = yaml_opt["results_dir"],
                    _additional_args = additional_args,
                    _GBT_alpha = _GBT_alpha,
                    DIST_TYPE = DIST_TYPE,
                        )
                f.write(command+"\nsleep 2\n")

if __name__ == '__main__':
    exps={}
    exps["dataset"]={"Bios_gender"}
    exps["batch_size"]={1024}
    exps["learning_rate"]={0.003}
    exps["hidden_size"]={300}
    exps["n_hidden"]={2}
    exps["dropout"]={0}
    exps["batch_norm"]={False}
    exps["random_seed"]=set([i for i in range(5)])
    exps["project_dir"]={"Vanilla"}
    exps["_GBT_alpha"] = [0.0, 0.25, 0.5, 0.75, 1, -0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    allNames=exps.keys()

    _file_path = "dist_bios_{}3/".format(DIST_TYPE)

    # # Vanilla
    # write_to_batch_files(job_name="Vanilla_"+DIST_TYPE, exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=[""])

    # DS
    DS_additional_args_list = ["--BT Downsampling --BTObj stratified_y"]
    write_to_batch_files(job_name="DS_"+DIST_TYPE, exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=DS_additional_args_list)

    # RW
    RW_additional_args_list = ["--BT Reweighting --BTObj stratified_y"]
    write_to_batch_files(job_name="RW_"+DIST_TYPE, exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=RW_additional_args_list)

    # # INLP
    # INLP_additional_args_list = ["--INLP --INLP_n 300 --INLP_by_class --INLP_discriminator_reweighting balanced --INLP_min_acc 0.5"]
    # write_to_batch_files(job_name="INLP_"+DIST_TYPE, exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=INLP_additional_args_list)

    # # Adv
    # Adv_additional_args_list = ["--adv_debiasing --adv_lambda {_adv_lambda} --adv_num_subDiscriminator 1 --adv_diverse_lambda 0".format(
    #     _adv_lambda = _adv_lambda
    # ) for _adv_lambda in set(log_grid(-1,1,20))]

    # write_to_batch_files(job_name="Adv_{}_{}".format(0,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=Adv_additional_args_list[:10])
    # write_to_batch_files(job_name="Adv_{}_{}".format(1,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=Adv_additional_args_list[10:])


    # # DAdv
    # DAdv_additional_args_list = ["--adv_debiasing --adv_lambda {_adv_lambda} --adv_num_subDiscriminator 3 --adv_diverse_lambda {_adv_diverse_lambda}".format(
    #     _adv_lambda = _adv_lambda,
    #     _adv_diverse_lambda = 0.1,
    # ) for _adv_lambda in set(log_grid(-1,1,20))]

    # write_to_batch_files(job_name="DAdv_{}_{}".format(0,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=DAdv_additional_args_list[:10])
    # write_to_batch_files(job_name="DAdv_{}_{}".format(1,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=DAdv_additional_args_list[10:])

    # # FairBatch
    # FairBatch_additional_args_list = ["--DyBT FairBatch --DyBTObj stratified_y --DyBTalpha {_DyBTalpha}".format(
    #     _DyBTalpha = _DyBTalpha
    # ) for _DyBTalpha in set(log_grid(-2,0,20))]

    # write_to_batch_files(job_name="FairBatch_{}_{}".format(0,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=FairBatch_additional_args_list[:10])
    # write_to_batch_files(job_name="FairBatch_{}_{}".format(1,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=FairBatch_additional_args_list[10:])


    # # EO_CLA
    # EOCla_additional_args_list = ["--DyBT GroupDifference --DyBTObj EO --DyBTalpha {_DyBTalpha}".format(
    #     _DyBTalpha = _DyBTalpha
    # ) for _DyBTalpha in set(log_grid(-3,-1,20))]

    # write_to_batch_files(job_name="EOCla_{}_{}".format(0,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=EOCla_additional_args_list[:10])
    # write_to_batch_files(job_name="EOCla_{}_{}".format(1,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=EOCla_additional_args_list[10:])

    # # EO_GLB
    # EOGlb_additional_args_list = ["--DyBT GroupDifference --DyBTObj joint --DyBTalpha {_DyBTalpha}".format(
    #     _DyBTalpha = _DyBTalpha
    # ) for _DyBTalpha in set(log_grid(-4,-2,20))]

    # write_to_batch_files(job_name="EOGlb_{}_{}".format(0,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=EOGlb_additional_args_list[:10])
    # write_to_batch_files(job_name="EOGlb_{}_{}".format(1,DIST_TYPE), exps=exps, allNames=allNames, file_path=_file_path, additional_args_list=EOGlb_additional_args_list[10:])