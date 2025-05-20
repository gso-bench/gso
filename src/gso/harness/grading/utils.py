from gso.utils.io import load_gso_dataset
from gso.constants import RUN_EVALUATION_LOG_DIR


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    reformat_reports: bool,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_gso_dataset(dataset_name, split)
    dataset_ids = {i.instance_id for i in dataset}
    print(f"Loaded {len(dataset)} instances from {dataset_name}/{split}.")

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(
                f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
            )

    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        print(
            f"Warning: Filtering {len(prediction_ids - dataset_ids)} predictions not in dataset."
        )
        prediction_ids = prediction_ids.intersection(dataset_ids)
        predictions = {k: v for k, v in predictions.items() if k in prediction_ids}

    if instance_ids:
        dataset = [i for i in dataset if i.instance_id in instance_ids]

    if reformat_reports:
        # we only return instances that have existing test outputs
        test_output_ids = set()
        for instance in dataset:
            if instance.instance_id not in predictions:
                continue
            prediction = predictions[instance.instance_id]
            test_output_file = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / prediction["model_name_or_path"].replace("/", "__")
                / prediction["instance_id"]
                / "test_output.txt"
            )
            if test_output_file.exists():
                test_output_ids.add(instance.instance_id)
        dataset = [
            i
            for i in dataset
            if i.instance_id in prediction_ids and i.instance_id in test_output_ids
        ]
        return predictions, dataset

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance.instance_id not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance.instance_id]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction["instance_id"]
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance.instance_id)

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i.instance_id not in completed_ids]

    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v["model_patch"] == "" or v["model_patch"] is None
    }
    print(f"Found {len(empty_patch_ids)} instances with empty patches.")

    # filter dataset to only instances with predictions
    dataset = [
        i
        for i in dataset
        if i.instance_id in prediction_ids and i.instance_id not in empty_patch_ids
    ]
    return predictions, dataset
