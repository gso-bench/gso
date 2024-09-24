import json
import os
from flask import Flask, request, render_template
from fuzzywuzzy import fuzz

from utils import load_data, get_test_status, get_function_name

json_file_path = "/home/manishs/buckets/pyperf_bucket/testgen/mpi4py_out.json"
data = load_data(json_file_path)
app = Flask(__name__)


@app.route("/")
def index():
    try:
        # Calculate overall stats
        total_functions = len(data)
        successful_tests = sum(1 for item in data if get_test_status(item) == "success")
        failed_tests = sum(1 for item in data if get_test_status(item) == "failure")
        timeouts = sum(1 for item in data if get_test_status(item) == "timeout")
        errors = sum(1 for item in data if get_test_status(item) == "error")

        # Apply filters
        filter_type = request.args.get("filter", "all")
        remove_tests = request.args.get("remove_tests", "false").lower() == "true"
        search_query = request.args.get("search", "").lower()

        def fid(x):
            return x.get("function_id", x.get("method_id", {})).get("identifier", "")

        filtered_data = data

        # Apply main filter
        if filter_type != "all":
            filtered_data = [
                item for item in filtered_data if get_test_status(item) == filter_type
            ]

        # Apply non-test filter if selected
        if remove_tests:
            filtered_data = [
                item for item in filtered_data if "test" not in fid(item).lower()
            ]

        # Apply search
        if search_query:
            filtered_data = sorted(
                [item for item in filtered_data if search_query in fid(item).lower()],
                key=lambda x: fuzz.partial_ratio(search_query, fid(x).lower()),
                reverse=True,
            )

        # Pagination
        page = int(request.args.get("page", 1))
        if page < 1 or page > len(filtered_data):
            page = 1

        current_item = filtered_data[page - 1] if filtered_data else None

        return render_template(
            "base.html",
            item=current_item,
            status=get_test_status(current_item) if current_item else None,
            total_functions=total_functions,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            errors=errors,
            timeouts=timeouts,
            search_query=search_query,
            filter_type=filter_type,
            remove_tests=remove_tests,
            current_page=page,
            total_pages=len(filtered_data),
        )
    except json.JSONDecodeError:
        return "Invalid JSON file.", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5600
    app.run(debug=True, host=host, port=port)
