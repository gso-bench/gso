import json
import os


def load_data(json_file_path):
    with open(json_file_path, "r") as fp:
        temp = json.load(fp)
        temp = [
            {key: value for key, value in item.items() if key != "context"}
            for item in temp
        ]

        # write this to temp.json in the same directory as json_file_path
        dir_path = os.path.dirname(json_file_path)

    temp_json_file_path = os.path.join(dir_path, "temp.json")
    with open(temp_json_file_path, "w") as temp_file:
        json.dump(temp, temp_file, indent=4)

    del temp

    # open temp.json and read the data
    with open(temp_json_file_path, "r") as temp_file:
        data = json.load(temp_file)

    # data = data[:5]

    return data


def get_test_status(item):
    if item["test_history"]["history"][0]["exec_stats"].get("error"):
        error_msg = item["test_history"]["history"][0]["exec_stats"]["error"]
        if "TimeoutError('result expired')" in error_msg:
            return "timeout"
        return "error"

    elif item["test_history"]["history"][0]["exec_stats"]["run_tests_logs"].get(
        "test_0"
    ):
        if (
            item["test_history"]["history"][0]["exec_stats"]["run_tests_logs"][
                "test_0"
            ]["passed_count"]
            > 0
        ):
            return "success"
        else:
            return "failure"
    else:
        return "error"


def get_function_name(item):
    return item.get("function_id", {}).get("identifier") or item.get(
        "method_id", {}
    ).get("identifier")
