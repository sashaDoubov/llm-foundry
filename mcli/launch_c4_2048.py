from mcli.sdk import RunConfig, create_run
import copy
import math

# or create directly in python with RunConfig(...)
config = RunConfig.from_file('mcli-base-train-for-mup-c4-d-model-2048.yaml')
parameters = copy.deepcopy(config.parameters)

# Sweep over a few different values of 'foo'
runs = []
base_lr = 1.e-3
c4_data = "oci://mosaicml-internal-datasets/c4/base/pretok-gpt2-2k/"
s2_data = "oci://mosaicml-internal-datasets/s2/base/pretok-gpt2-2k/"


load_paths =[
    "mpt-40m-d-2048-lr-0-0005-data-c4-e-scale-1-Kzk83F",
    "mpt-40m-d-2048-lr-0-0005-data-c4-e-scale-10-0TEbBB",
    "mpt-40m-d-2048-lr-0-001-data-c4-e-scale-1-au1LNf",
    "mpt-40m-d-2048-lr-0-001-data-c4-e-scale-10-tToQzR",
    "mpt-40m-d-2048-lr-0-002-data-c4-e-scale-1-7m0zd1",
    "mpt-40m-d-2048-lr-0-002-data-c4-e-scale-10-U7GnVu",
    "mpt-40m-d-2048-lr-0-004-data-c4-e-scale-1-6F3hLj",
    "mpt-40m-d-2048-lr-0-004-data-c4-e-scale-10-Rzs0MV",
]

for embed_scale in [1.0, 10.0]:
    for friendly_data, dataset in zip(["s2"], [s2_data]):
        for lr_scaler in [-1, 0]:
        # for lr_scaler in [0]:
            # set the name of the run
            updated_lr = base_lr * (2**lr_scaler)
            print(updated_lr)
            print(dataset)

            rounded_num = round(updated_lr, 2 - int(math.floor(math.log10(abs(updated_lr)))) - 1)
            print(rounded_num)

            convert_to_string = str(updated_lr).replace(".", "-")
            config.name = f'mpt-40m-d-2048-lr-{convert_to_string}-data-{friendly_data}-e-scale-{int(embed_scale)}'

            print(config.name)

            # for s in load_paths:
            #     # add end dash to avoid matching 10 with 1
            #    if s.startswith(config.name + "-"):
            #        load_folder_path = s
            #        break
            # else:
            #     raise Exception("load path not found!")

            # load_path = f"oci://mosaicml-internal-checkpoints/sasha/large_run_sweep/{load_folder_path}/checkpoints/latest-rank0.pt"

            # print(load_path)

            # Update the parameters
            # deepcopy for safety
            run_params = copy.deepcopy(parameters)
            run_params['optimizer']["lr"] = updated_lr
            run_params['optimizer']["weight_decay"] = updated_lr
            run_params['data_remote'] = dataset
            run_params["model"]["mup"]["embed_scale"] = embed_scale

            run_params["eval_subset_num_batches"] = 332
            # run_params["load_path"] = load_path

            if friendly_data == "s2":
                # change to be same as c4
                run_params["eval_subset_num_batches"] = 332
            config.parameters = run_params
            print(config.parameters)

            # And run!
            run = create_run(config)
            print(f'Launching run {run.name} with lr {updated_lr} and data {friendly_data} and embed scale {embed_scale}')
            runs.append(run)