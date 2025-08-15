import warnings

warnings.filterwarnings(
    "ignore",
    message="BigQuery Storage module not found, fetch data with the REST endpoint instead.",
)

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).",
)
