import csv

def remove_columns(input_path: str, output_path: str, columns_to_remove: list[str]) -> None:
    """
    Reads a CSV file, removes specified columns, and writes to a new CSV file.
    """
    remove_set = set(columns_to_remove)

    with open(input_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_path, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        # Filter header columns
        if reader.fieldnames is None:
            raise ValueError("The input CSV file appears to be empty.")

        filtered_fieldnames = [field for field in reader.fieldnames if field not in remove_set]

        writer = csv.DictWriter(outfile, fieldnames=filtered_fieldnames)
        writer.writeheader()

        # Write rows without the specified columns
        for row in reader:
            filtered_row = {k: v for k, v in row.items() if k not in remove_set}
            writer.writerow(filtered_row)

# Example Usage:
cols_to_drop = [
    'employee_id',
    'age',
    'gender',
    'marital_status',
    'education_level',
    'department',
    'job_role',
    'job_level',
    'monthly_income',
    'stock_option_level',
    'salary_hike_pct',
    'years_at_company',
    'total_working_years',
    'num_prior_companies',
    'work_mode',
    'commute_minutes',
    'business_travel_days_per_year',
    'training_hours',
    'performance_rating',
    'last_review_score',
    'uses_ai_tools_at_work',
    'perceived_ai_job_risk'
]
remove_columns('employee_attrition_hr_2026.csv', 'dataset.csv', cols_to_drop)