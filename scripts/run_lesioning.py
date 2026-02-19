from facebody.lesioning import run_lesioning_analysis

device = "cuda:0"

# Specify model, layer, and tasks
model_name = "alexnet_ecoset"
layer = "relu7"
tasks = ["face", "person", "action"]

# Run lesioning analysis
for task in tasks:
    run_lesioning_analysis(
        model_name, readout_layer=layer, device=device, task=task,
    )
